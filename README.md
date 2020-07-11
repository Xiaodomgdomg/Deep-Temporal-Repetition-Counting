# CVPR2020-Deep-Temporal-Repetition-Counting
This code is implemented based on the project["3D ResNets for Action Recognition"](https://github.com/kenshohara/3D-ResNets-PyTorch). 

## Requirements

* [PyTorch](http://pytorch.org/) (ver. 1.0)
* Python 2

## Dataset Preparation

### UCFRep
* Please download the UCF101 dataset [here](http://crcv.ucf.edu/data/UCF101.php).
 * Convert UCF101 videos from avi to png files, put the png files to data/ori_data/ucf526/imgs/train
 * Create soft link with following commands:
```bash
cd data/ori_data/ucf526/imgs
ln -s train val
```
* Please download the anotations ([Google Drive](https://drive.google.com/file/d/1c0v51oP44lY_PhpJp8KYAwDaQmxj2zcs/view?usp=sharing),or [Baidu Netdisk](https://pan.baidu.com/s/1nHQZ8P-JZPTo4IRlcOBoHA) code:n5za), and put it to data/ori_data/ucf526/annotations

### QUVA
* Please download the QUVA dataset in: http://tomrunia.github.io/projects/repetition/
 * Put the label files to data/ori_data/QUVA/annotations/val
 * Convert QUVA videos to png files, put the png files to data/ori_data/QUVA/imgs
 
### YTsegments
* Please download the YTSeg dataset in: https://github.com/ofirlevy/repcount 
 * Put the label files to data/ori_data/YT_seg/annotations
 * Convert YTsegments videos to png files, put the png files to data/ori_data/YT_seg/imgs

## Running the code
### Training
Train from scratch
```bash
python main.py
```
If you want to finetune the model pretrained on Kinetics, first you need to download the pretrained model in [here](https://github.com/kenshohara/3D-ResNets-PyTorch) and run:
```bash
python main.py --pretrain_path = pretrained_model_path
```

### Testing
You can also run the trained model provide by ours ([Google Drive]()):
```bash
python main.py --no_train --resume_path = trained_model_path
```

## Citation
If you use this code or pre-trained models, please cite the following:

```bibtex
 @InProceedings{Zhang_2020_CVPR,
    author = {Zhang, Huaidong and Xu, Xuemiao and Han, Guoqiang and He, Shengfeng},
    title = {Context-Aware and Scale-Insensitive Temporal Repetition Counting},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
} 
```



