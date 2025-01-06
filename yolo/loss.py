import torch

from torchvision.ops import box_iou

class YOLOLoss(object):
    def __init__(self):
        """
        Loss terms:
        1. MSE for bounding box (size = 1)
        2. BCE for confidence score / IoU (size = 1)
        3. BCE for classification for class (size = # of classes)
        """
        self.obj_bce = torch.nn.BCEWithLogitsLoss()
        self.class_bce = torch.nn.BCEWithLogitsLoss()

    def __call__(self, preds, labels):
        """ 
        preds: list of length 3, each element corresponds to prediction at
        a particular scale, with shape:

            (bs, na, nx, ny, 5 + nc)

        where
        - bs: batch size
        - na: anchor index
        - nx, ny: cell index
        - nc: number of classes

        the ordering of the 5 bounding box + class prediction is: [x, y, w, h, c]

        labels: a tensor with shape:

            (bs, m, 5)

        where
        - bs: batch size
        - m: number of annotated objects in an image
        - last dimension: class c and bounding box, [c, x, y, w, h]

        """
        for p in preds:
            pass