# yolo-v3-finetune

## Overview
Finetuning YOLO V3 with the `lvis` [dataset](https://www.lvisdataset.org/).

In order to extend the prediction vocabulary of the basic model pretrained with `coco` dataset. Based on the same set of images, the `lvis` dataset is based on contains 1203 annotated classes compared to just 80 classes in `coco`.

The underlying model and weights are from Torch Hub: https://pytorch.org/hub/ultralytics_yolov5/ uploaded by Ultralytics.

Apart from that, this repository contains my own implementation of a dataloader, loss function and evaluation/training code packaged with [`lightning`](https://lightning.ai/docs/pytorch/stable/). 

## Setup
Prequisites: install `pyenv` and install python version `> 3.10`
```
poetry install
```
TODO: 

## Train/evaluation
WIP

## References
Here are various references I found helpful while working on the code :)

- YOLO papers [1](https://arxiv.org/pdf/1506.02640), [2](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
- [C4W3L09 YOLO Algorithm
](https://www.youtube.com/watch?v=9s_FpMpdYW8)
- [What’s new in YOLO v3?](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)
- https://github.com/w86763777/pytorch-simple-yolov3

