import torch

from yolo.loss_utils import (
    get_box_coordinates, 
    obj_anchors_ious, 
    make_grid_idx, 
    make_one_hot_label, 
    standardize_label_box,
    compute_prediction_mask
)

torch.manual_seed(0)


def test_loss_get_box(pred_index_shape, image_size, anchors, stride):
    pred_box = torch.rand(size=(*pred_index_shape, 4))
    pred_box = pred_box * image_size
    box = get_box_coordinates(pred_box, anchors, stride)
    assert box.shape == pred_box.shape


def test_standardize_label_box(image_size, num_objects, anchors, stride):
    label_box = torch.rand(size=(num_objects, 4)) * image_size
    best_idx = torch.randint(low=0, high=len(anchors), size=(num_objects, 1))
    box = standardize_label_box(label_box, best_idx, anchors, stride)
    assert box.shape == label_box.shape


def test_compute_prediction_mask(pred_index_shape, image_size, num_objects, ignore_threshold):
    pred_box = torch.rand(size=(*pred_index_shape, 4))
    pred_box = pred_box * image_size
    label_box = torch.rand(size=(num_objects, 4))
    mask = compute_prediction_mask(pred_box, label_box, ignore_threshold)
    assert mask.shape == pred_index_shape


def test_obj_anchors_ious(anchors, image_size, num_objects):
    label_box = torch.rand(size=(num_objects, 4)) * image_size
    ious = obj_anchors_ious(label_box, anchors)
    assert ious.shape == (num_objects, len(anchors))


def test_make_one_hot_label(num_objects, num_classes):
    label = torch.randint(
        low=0, 
        high=num_classes, 
        size=(num_objects, 4)
    )
    one_hot = make_one_hot_label(label, num_classes)
    assert one_hot.shape == (num_objects, num_classes)


def test_make_grid_idx(num_objects, image_size, stride):
    label = torch.rand(size=(num_objects, 5)) * image_size
    grid_idx = make_grid_idx(label, stride)
    assert grid_idx.shape == (num_objects, 2)