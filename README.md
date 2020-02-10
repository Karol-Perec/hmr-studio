### Prepare for training:

Copy unzipped files from https://drive.google.com/file/d/1X2XCYFGMRxqAc4bnF83nGLmKEW1rO6nr/view?usp=sharing to:
```
models/
```
Files structure as below:
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
Files structure as below:
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

### Imdb wiki dataset preparation

Copy unzipped files from https://drive.google.com/file/d/1JcwYNXFX__HIJ-JSoLNerucfvDq3YFsQ/view?fbclid=IwAR1oFhAfmFfad88orlG68tIqv4KvAX8K5IaEH9NGtdD2k2L2_WaqCJ1ryIg to:
```
datasets/human/imdb
```
Files structure as below:
```
datasets/
├── tf_datasets
└── human
    └── imdb
        ├── result.mat
        └── images
            ├── 0000.jpg
            ├── ...
            └── 1641.jpg

```
Run:
```
prepare_datasets.sh
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
