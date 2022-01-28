import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import densenet161
import torchvision.transforms as transforms

from .focal import FocalTransformer


class DepthModel(nn.Module):
    def __init__(self, dataset = 'NYU', pretrained = False):
        super(DepthModel, self).__init__()
        if dataset == 'NYU':
            self.max_depth = 10.0
            self.out_size = [416,544]
        elif dataset == 'KITTI':
            self.max_depth = 80.0
            self.out_size = [352,704]
        self.vit =  FocalTransformer(img_size=224, embed_dim=128, depths=[2,2,18,2], drop_path_rate=0.2, 
                                    focal_levels=[2,2,2,2], expand_sizes=[3,3,3,3], expand_layer="all", 
                                    num_heads=[4,8,16,32],
                                    focal_windows=[7,5,3,1], 
                                    window_size=7,
                                    use_conv_embed=True, 
                                    use_shift=False, 
                                    focal_pool='fc',
                                    focal_topK=128,
                                    focal_stages=[0,1,2,3])
        if pretrained:
            self.vit.load_state_dict(torch.load('focalv2-base-useconv-is224-ws7.pth')['model'], strict=True)
        self.denseNet = DenseNet(pretrained = pretrained)
        self.up1 = UpSample(2600, 512)
        self.up2 = UpSample(2722, 256)
        self.up3 = UpSample(1073, 128)
        self.up4 = UpSample(576, 128)
        self.greyconv = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(32),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.LeakyReLU())

        self.final_conv = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(32),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(3),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(3,  1, kernel_size=3, stride=1, padding=1))




    def forward(self, x224, x512):
        vx64, vx32, vx16, _, vx8 = self.vit(x224)
        dx128, dx64, dx32, dx16 = self.denseNet(x512)


        x = self.up1(vx16.view(-1, 196, 16, 16), dx16, vx8.view(-1, 196, 16, 16))
        x = self.up2(vx32.view(-1, 98, 32, 32), x, dx32)
        x = self.up3(vx64.view(-1, 49, 64, 64), x, dx64)

        x224x = F.interpolate(transforms.Grayscale()(x224), size=[128,128], mode='bilinear', align_corners=True)
        x128x = self.greyconv(x224x)

        x = self.up4(x128x, x, dx128)

        x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=True)
        x = self.final_conv(x)

        x = torch.sigmoid(x) * self.max_depth

        return x



class DenseNet(torch.nn.Module):
    def __init__(self, pretrained):
        super(DenseNet, self).__init__()
        features = list(densenet161(pretrained = pretrained).features)
        self.features = torch.nn.ModuleList(features)
        for param in self.features[:7].parameters():
          param.requires_grad = False

    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {4,6,8,10}:
                results.append(x)
          
        return results


class UpSample(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, one, two, three):
        x = torch.cat([one, two, three], dim=1)
        x = self._net(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x
