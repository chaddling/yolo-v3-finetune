import torch

from yolo.loss import YOLOLoss

def test_loss(
    num_classes, 
    stride, 
    image_size, 
    ignore_threshold, 
    pred_index_shape, 
    num_objects
):
    all_anchors = [
        (10, 13, 16, 30, 33, 23),
        (30, 61, 62, 45, 59, 119),
        (116, 90, 156, 198, 373, 326)
    ]
    anchor_idx = 1
    loss = YOLOLoss(
        num_classes,
        all_anchors,
        anchor_idx,
        stride,
        image_size,
        ignore_threshold,
    )

    preds = torch.rand((*pred_index_shape, num_classes + 5))
    batch_idx = torch.randint(
        low=0, 
        high=pred_index_shape[0], 
        size=(num_objects, 1)
    ).sort().values
    classes = torch.randint(
        low=0,
        high=5,
        size=(num_objects, 1)
    )
    label = torch.cat(
        [
            batch_idx,
            classes,
            torch.rand(size=(num_objects, 4)) * image_size
        ],
        dim=-1
    )
    result = loss(preds, label)
    assert result.shape == ()
