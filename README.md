# ResViT
Official Pytorch Implementation of Residual Vision Transformers(ResViT) which is described in the [following](https://arxiv.org/abs/2106.16031) paper:

Onat Dalmaz and Mahmut Yurt and Tolga Çukur ResViT: Residual vision transformers for multi-modal medical image synthesis. arXiv. 2021.

<img src="main_fig.png" width="600px"/>

## Dependencies

```
python>=3.6.9
torch>=1.7.1
torchvision>=0.8.2
visdom
dominate
cuda=>11.2
```

## Download pre-trained ViT models from Google
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/R50-ViT-B_16.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/R50-ViT-B_16.npz
```

## Dataset
You should structure your aligned dataset in the following way:
```
/Datasets/BRATS/
  ├── T1_T2
  ├── T2_FLAIR
  .
  .
  ├── T1_FLAIR_T2   
```
```
/Datasets/BRATS/T2__FLAIR/
  ├── train
  ├── val  
  ├── test   
```

