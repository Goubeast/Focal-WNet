# Focal-WNet
Official PyTorch Code for Monocular Depth Estimation using Focal-WNet
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/focal-wnet-an-architecture-unifying/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=focal-wnet-an-architecture-unifying)

## Qualitative Result on NYU v2 Dataset

![Comparison](https://user-images.githubusercontent.com/59992424/151580639-8827e849-ed74-40a9-9a46-1d7f6802ef6e.png)

## Requirements
* Python >= 3.7
* Pytorch >= 1.6.0
*  CUDA 9.2
*  cuDNN (if CUDA available)

Rest of the libraries can be installed by ``` pip install -r requirements.txt ```

### Pretrained Models
*Will be uploaded shortly*

## Prepare Data
Please refer to [LapDepth](https://github.com/tjqansthd/LapDepth-release) for the dataset preparation process for [NYU v2 Depth](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and [KITTI](http://www.cvlibs.net/download.php?file=data_depth_annotated.zip)

## Training

```python
# KITTI 
python train.py --distributed --batch_size 16 --dataset KITTI --data_path ./datasets/KITTI --gpu_num 0,1,2,3
# NYU
python train.py --distributed --batch_size 16 --use_dense_depth --dataset NYU --data_path ./datasets/NYU_Depth_V2/sync --epochs 30 --gpu_num 0,1,2,3 
```
## Testing

```python
# KITTI ## KITTI
python eval.py --model_dir ./pretrained/LDRN_KITTI_ResNext101_pretrained_data.pkl --evaluate --batch_size 1 --dataset KITTI --data_path ./datasets/KITTI --gpu_num 0
## NYU Depth V2
python eval.py --model_dir ./pretrained/LDRN_NYU_ResNext101_pretrained_data.pkl --evaluate --batch_size 1 --dataset NYU --data_path --data_path ./datasets/NYU_Depth_V2/official_splits/test --gpu_num 0

```

## Reference 
We thank the authors of the following repositories for their contribution.

[LapDepth](https://github.com/tjqansthd/LapDepth-release)

[Focal Transformer](https://github.com/microsoft/Focal-Transformer)
