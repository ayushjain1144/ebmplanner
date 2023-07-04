import numpy as np
import torch

import ipdb

st = ipdb.set_trace


def _clamp_boxes(frame_boxes, H, W):
    frame_boxes = np.clip(frame_boxes, a_min=0, a_max=None)
    frame_boxes = np.minimum(
        frame_boxes, np.array([W-1, H-1, W-1, H-1]).reshape(1, 4))
    return frame_boxes


def clamp_boxes_torch(boxes, H, W):
    boxes = torch.clamp(boxes, min=0, max=None)
    boxes = torch.minimum(
        boxes, torch.tensor([W-1, H-1, W-1, H-1]).reshape(1, 4).to(boxes.device))
    return boxes


def _load_gt_boxes(detections, modes, H, W, flip=False):
    all = []
    for mode in modes:
        for k in list(detections[mode].values()):
            x1_, y1_, x2_, y2_ = k[0][0][1], k[0][0][0], k[0][1][1], k[0][1][0]
            box = np.array([x1_, y1_, x2_, y2_]).reshape(1, -1)
            box = _clamp_boxes(box, H, W).astype(np.int64).reshape(-1).tolist()
            x1_, y1_, x2_, y2_ = box
            if flip:
                box = np.array([y1_, x1_, y2_, x2_]).astype(np.int64).reshape(-1).tolist()
            else:
                box = np.array([x1_, y1_, x2_, y2_]).astype(np.int64).reshape(-1).tolist()
            all.append(box)
    return all


def flip_data(img, ground_truths):
    img = img.transpose(1, 0, 2)
    for i in range(len(ground_truths)):
        if type(ground_truths[i][0]) == list:
            for j in range(len(ground_truths[i])):
                x1, y1, x2, y2 = ground_truths[i][j]
                ground_truths[i][j] = [y1, x1, y2, x2]
        else:
            x1, y1, x2, y2 = ground_truths[i]
            ground_truths[i] = [y1, x1, y2, x2]
    return img, ground_truths


def flip_boxes(boxes):
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        boxes[i] = torch.tensor([y1, x1, y2, x2])
    return boxes


def flip_heatmaps(pick_hmap, place_hmap):
    pick_hmap = pick_hmap.permute(0, 2, 1)
    place_hmap = place_hmap.permute(0, 2, 1)

    return pick_hmap, place_hmap


def mask_image_with_boxes(image, boxes):
    # box is X1, Y1, X2, Y2
    # B X 4
    masked_image_ = torch.zeros_like(image)
    padding = 3 # untested

    boxes[:, :2] -= padding
    boxes[:, 2:] += padding

    image_size = image.shape
    boxes[:, (0, 2)] = torch.clamp(boxes[:, (0, 2)], 0, image_size[1])
    boxes[:, (1, 3)] = torch.clamp(boxes[:, (1, 3)], 0, image_size[0])

    for box in boxes:
        x1, y1, x2, y2 = box
        masked_image_[y1:y2, x1:x2] = image[y1:y2, x1:x2]

    return masked_image_


# this works for 640 X 320 image
def get_all_binary_relations(rel_box, ref_box):
    relations_ = []
    x1, y1, x2, y2 = rel_box
    x1_, y1_, x2_, y2_ = ref_box

    if x2 < x1_:
        relations_.append("left")
    if x1 > x2_:
        relations_.append("right")
    if y2 < y1_:
        relations_.append("above")
    if y1 > y2_:
        relations_.append("below")
    return relations_


# this works for 640 X 320 image
def execute_relation(boxes_, rel_id, ref_id, relation, image_size=[640, 320]):
    H, W = image_size
    # sample randomly center for new box
    rel_cx = np.random.choice(
        np.arange(0, H)
    )
    rel_cy =  np.random.choice(
        np.arange(0, W)
    )
    x1, y1, x2, y2 = boxes_[ref_id]

    buffer = 60 # pixels to avoid intersection
    if relation == "left":
        if x1 - buffer <= 0:
            return boxes_
        else:
            rel_cx = np.random.choice(
                np.arange(0, x1 - buffer)
            )
    elif relation == "right":
        if x2 + buffer >= H:
            return boxes_
        else:
            rel_cx = np.random.choice(
                np.arange(x2 + buffer, H)
            )
    elif relation == "above":
        if y1 - buffer <= 0:
            return boxes_
        else:
            rel_cy = np.random.choice(
                np.arange(0, y1 - buffer)
            )
    elif relation == "below":
        if y2 + buffer >= W:
            return boxes_
        else:
            rel_cy = np.random.choice(
                np.arange(y2 + buffer, W)
            )
    else:
        assert False, print(relation)

    # update box
    w, h = boxes_[rel_id][2:] - boxes_[rel_id][:2]
    boxes_[rel_id] = np.array([
        rel_cx - w / 2, 
        rel_cy - h / 2,
        rel_cx + w / 2,
        rel_cy + h / 2
    ])
    rels = get_all_binary_relations(
        boxes_[rel_id], boxes_[ref_id])
    if relation not in rels:
        st()
    # assert relation in rels
    return boxes_
