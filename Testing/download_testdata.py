!mkdir datasets
!wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat

!python extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ./NYU_Depth_V2/official_splits/