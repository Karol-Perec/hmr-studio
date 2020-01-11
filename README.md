### Prepare for training:

Copy unzipped files from https://drive.google.com/file/d/1X2XCYFGMRxqAc4bnF83nGLmKEW1rO6nr/view?usp=sharing to:
```
models/
```
Files structure as above:
```
models/
├── eval.graph
├── model.ckpt-667589.data-00000-of-00001
├── model.ckpt-667589.index
├── model.ckpt-667589.meta
├── neutral_smpl_mean_params.h5
├── neutral_smpl_meanwjoints.h5
├── neutral_smpl_with_cocoplus_reg.pkl
├── neutral_smpl_with_cocoplustoesankles_reg.pkl
├── resnet_v2_50.ckpt
└── train.graph
```
 
Copy unzipped files from https://drive.google.com/file/d/1Q7UJe04D5YMznCZgKvlgCPW-d4h3_RrY/view?usp=sharing to:
```
datasets/tf_datasets
```
Files structure as above:
```
datasets/
├── human
└── tf_datasets
    ├── coco
    ├── lsp
    ├── lsp_ext
    ├── mocap_neutrMosh
    ├── mpii
    └── mpi_inf_3dhp
```

### Training 

Just run:
```
do_train.sh
```

### Demo

Run the demo:
```
python -m demo --img_path data/coco1.png
python -m demo --img_path data/im1954.jpg
```
