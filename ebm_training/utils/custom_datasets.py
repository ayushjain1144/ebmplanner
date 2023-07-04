"""Dataset class for training EBMs on simple concepts."""

from copy import deepcopy
import json
import os
import os.path as osp

import numpy as np
import torch
from torch.utils.data import Dataset

from .ebm_data_generator import (
    create_graph_data,
    XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX, SIZE
)


class EBMDataset(Dataset):
    """Dataset utilities for concept learning."""

    def __init__(self, split='train', concept=['right'], n_samples=10):
        """Initialize dataset."""
        self.split = split
        self.annos = []
        self.concepts = concept
        for conc in concept:
            self.annos += self.load_annos(conc, n_samples)
        print('Loaded %d samples for %s' % (len(self.annos), concept))
        self.lens = np.array(
            [XMAX - XMIN - SIZE, YMAX - YMIN - SIZE, ZMAX - ZMIN - SIZE]
        ) / 2.0

        self.shape_list = [
            'line',
            'alpha',
            'circle',
            'triangle',
            'square',
            'racer'
        ]
        self.shapes = {shape: s for s, shape in enumerate(self.shape_list)}

        self.rel_list = [
            'front',
            'behind',
            'left',
            'right',
            'inside',
            'supported-by'
        ]
        self.relations = {rel: r for r, rel in enumerate(self.rel_list)}

    def load_annos(self, concept, n_samples):
        """
        Load annotations.

        {
            'label': label,
            'ref_boxes': ref_boxes.tolist(),  # boxes of reference
            'rel_boxes': boxes.tolist(),  # in relation with reference
            'attention': attention_inds.tolist()  # map to rel_boxes
        }
        """
        if concept == 'pose':
            s = {'label': 'pose'}
            return [s] * 12000
        split = 'train' if self.split == 'train' else 'test'
        os.makedirs('data/', exist_ok=True)
        if not osp.exists(f'data/{concept}_{n_samples}.json'):
            create_graph_data(
                f'data/{concept}_{n_samples}.json',
                concept,
                2 * n_samples
            )
        with open(f'data/{concept}_{n_samples}.json') as fid:
            annos = json.load(fid)[split]
        # return annos
        data = np.copy(annos)
        annos = []
        for _ in range(int(500 * 24 / n_samples)):
            annos += data.tolist()
        return annos

    def __getitem__(self, index):
        """Get current batch for input index."""
        anno = deepcopy(self.annos[index])
        label = anno['label']
        if label == 'pose':
            return self._get_pose(None)
        if label in ('square', 'circle', 'line', 'triangle', 'alpha'):
            return self._get_shape(anno)
        if label == 'supported-by':
            return self._get_relation3d(anno)
        if label == 'racer':
            return self._get_posed_shape(anno)
        return self._get_relation(anno)

    def _get_relation(self, anno):
        # Fill missing values
        t = np.array(anno['attention']).argmax()
        rel_boxes = [anno['rel_boxes'][t]]
        # Convert boxes to 'points' (useful for augmentation)
        boxes = np.asarray(
            anno['ref_boxes'] + rel_boxes + [anno['rel_boxes'][t - 1]]
        )
        boxes = np.concatenate((
            boxes[:, :3] - boxes[:, 3:] / 2, boxes[:, :3] + boxes[:, 3:] / 2
        ), 1)
        boxes = boxes.reshape(len(boxes) * 2, 3)
        # Augmentation
        if self.split == 'train':
            # boxes, _ = self._augment_rot(anno['label'], boxes)
            boxes = self._augment_pos(boxes)
            boxes = self._augment_scale(anno['label'], boxes)
        # Back to boxes (x_c, y_c, z_c, W, D, H)
        boxes = boxes.reshape(-1, 6)
        centers = (boxes[:, :3] + boxes[:, 3:]) / 2
        size = np.abs(boxes[:, 3:] - boxes[:, :3])
        boxes = np.concatenate((centers[..., :2], size[..., :2]), 1)
        boxes[-1, 2:] = boxes[1, 2:]
        boxes[-1] = self._make_neg(boxes[-1], boxes[0], anno['label'])
        return {
            "label": self.relations[anno['label']],
            "sboxes": torch.from_numpy(boxes[1]).float(),
            "oboxes": torch.from_numpy(boxes[0]).float(),
            "noisy_sboxes": torch.from_numpy(boxes[2]).float()
        }

    def _get_relation3d(self, anno):
        # Fill missing values
        t = np.array(anno['attention']).argmax()
        rel_boxes = [anno['rel_boxes'][t]]
        # Convert boxes to 'points' (useful for augmentation)
        boxes = np.asarray(
            anno['ref_boxes'] + rel_boxes + [anno['rel_boxes'][t - 1]]
        )
        boxes = np.concatenate((
            boxes[:, :3] - boxes[:, 3:] / 2, boxes[:, :3] + boxes[:, 3:] / 2
        ), 1)
        boxes = boxes.reshape(len(boxes) * 2, 3)
        # Augmentation
        if self.split == 'train':
            boxes, _ = self._augment_rot(anno['label'], boxes)
            boxes = self._augment_pos(boxes)
            boxes = self._augment_scale(anno['label'], boxes)
        # Back to boxes (x_c, y_c, z_c, W, D, H)
        boxes = boxes.reshape(-1, 6)
        centers = (boxes[:, :3] + boxes[:, 3:]) / 2
        size = np.abs(boxes[:, 3:] - boxes[:, :3])
        boxes = np.concatenate((centers[..., :3], size[..., :3]), 1)
        boxes[-1, 3:] = boxes[1, 3:]
        boxes[-1, 0] = np.random.uniform(XMIN, XMAX, 1)
        boxes[-1, 1] = np.random.uniform(YMIN, YMAX, 1)
        boxes[-1, 2] = np.random.uniform(ZMIN, ZMAX, 1)
        return {
            "label": self.relations[anno['label']],
            "sboxes": torch.from_numpy(boxes[1]).float(),
            "oboxes": torch.from_numpy(boxes[0]).float(),
            "noisy_sboxes": torch.from_numpy(boxes[2]).float()
        }

    def _get_shape(self, anno):
        # Fill missing values
        t = np.array(anno['attention'])
        boxes = np.array(anno['ref_boxes'])[t > 0]
        # Convert boxes to 'points' (useful for augmentation)
        boxes = np.concatenate((
            boxes[:, :3] - boxes[:, 3:] / 2, boxes[:, :3] + boxes[:, 3:] / 2
        ), 1)
        boxes = boxes.reshape(len(boxes) * 2, 3)
        # Augmentation
        if self.split == 'train' or True:
            boxes, _ = self._augment_rot(anno['label'], boxes)
            boxes = self._augment_pos(boxes)
            boxes = self._augment_scale(anno['label'], boxes)
        # Back to boxes (x_c, y_c, z_c, W, D, H)
        boxes = boxes.reshape(-1, 6)
        centers = (boxes[:, :3] + boxes[:, 3:]) / 2
        size = np.abs(boxes[:, 3:] - boxes[:, :3])
        # Convert to 2d
        boxes = np.concatenate((centers[..., :2], size[..., :2]), 1)
        neg_boxes = np.copy(boxes)
        neg_boxes[:, 0] = np.random.uniform(XMIN, XMAX, (len(neg_boxes),))
        neg_boxes[:, 1] = np.random.uniform(YMIN, YMAX, (len(neg_boxes),))
        # Radius
        if anno['label'] == 'circle':
            centers = boxes[..., :2]
            centroid = centers.mean(0)
            radius = np.sqrt(np.sum((centers - centroid[None]) ** 2, 1)).mean()
            centers = centers / radius * (0.09 * np.random.rand(1)[0] + 0.13)
            boxes[:, :2] = centers
            r = np.random.rand(1)[0]
            if np.random.rand(1)[0] > 0.7:
                r = 10
            neg_boxes[:, :2] = (r * neg_boxes[:, :2] + boxes[:, :2]) / (r + 1)
        # Padding
        pad_boxes = np.zeros((10, 4))
        pad_neg_boxes = np.zeros((10, 4))
        pad_att = np.zeros(10)
        pad_boxes[:len(boxes)] = boxes
        pad_neg_boxes[:len(neg_boxes)] = neg_boxes
        pad_att[:len(boxes)] = 1
        return {
            "sboxes": torch.from_numpy(pad_boxes).float(),
            "noisy_sboxes": torch.from_numpy(pad_neg_boxes).float(),
            "attention": torch.from_numpy(pad_att),
            "label": int(self.shapes[anno["label"]])
        }

    def _get_posed_shape(self, anno):
        # Augmentation
        centers = np.array(anno['points'])
        xmin, ymin = centers.min(0).flatten().tolist()
        xmax, ymax = centers.max(0).flatten().tolist()
        x_tr = np.random.uniform(-xmin - self.lens[0], self.lens[0] - xmax, 1)
        y_tr = np.random.uniform(-ymin - self.lens[1], self.lens[1] - ymax, 1)
        centers[:, 0] += x_tr
        centers[:, 1] += y_tr
        centers = self._augment_scale(anno['label'], centers)
        return {
            "points": torch.from_numpy(centers).float(),
            "neg_points": torch.from_numpy(
                np.random.uniform(-0.85, 0.85, (6, 2))
            ).float(),
            "theta": torch.as_tensor(anno['angles']) / np.pi,
            "neg_theta": torch.as_tensor(
                np.random.uniform(-np.pi, np.pi, (6,))
            ) / np.pi,
            "label": int(self.shapes[anno["label"]])
        }

    def _get_pose(self, index):
        point = np.random.uniform(-0.85, 0.85, (2,))
        target = np.random.uniform(-0.85, 0.85, (2,))
        theta = np.arctan2(target[1] - point[1], target[0] - point[0])
        theta = np.array([theta]) / np.pi
        neg_theta = np.random.uniform(-np.pi, np.pi, (1,)) / np.pi
        return {
            "points": torch.from_numpy(point).float(),
            "targets": torch.from_numpy(target).float(),
            "theta": torch.from_numpy(theta).float(),
            "neg_theta": torch.from_numpy(neg_theta).float()
        }

    def _make_neg(self, box, ref_box, rel):
        box[0] = np.random.uniform(XMIN, XMAX, 1)
        box[1] = np.random.uniform(YMIN, YMAX, 1)
        if rel == 'front':
            ref_y = ref_box[1] - ref_box[3] * 0.5 - box[3] * 0.5
            box[1] = np.random.uniform(ref_y, YMAX, 1)
        if rel == 'left':
            ref_x = ref_box[0] - ref_box[2] * 0.5 - box[2] * 0.5
            box[0] = np.random.uniform(ref_x, XMAX, 1)
        if rel == 'behind':
            ref_y = ref_box[1] + ref_box[3] * 0.5 + box[3] * 0.5
            box[1] = np.random.uniform(YMIN, ref_y, 1)
        if rel == 'right':
            ref_x = ref_box[0] + ref_box[2] * 0.5 + box[2] * 0.5
            box[0] = np.random.uniform(XMIN, ref_x, 1)
        return box

    def __len__(self):
        """Return number of samples."""
        return len(self.annos)

    @staticmethod
    def _augment_rot(label, boxes):
        if label in ('square', 'circle', 'line', 'cube', 'triangle', 'alpha'):
            max_degrees = (0, 0, 180)
            if label in ('line', 'square'):
                max_degrees = (0, 0, 0)
        elif label in ('pizza', 'racer', 'caravan'):
            max_degrees = (0, 0, 180)
        elif label in ('left', 'right'):
            max_degrees = (0, 0, 3)  # (180, 3, 3)
        elif label in ('front', 'behind'):
            max_degrees = (0, 0, 3)  # (3, 180, 3)
        elif label in ('above', 'below', 'supported-by', 'supporting'):
            max_degrees = (0, 0, 180)
        else:
            max_degrees = (180, 180, 180)
        random_rots = [0, 0, 0]
        for i in range(3):
            random_rots[i] = (2 * np.random.rand() - 1) * max_degrees[i]
        if label == 'line':
            random_rots = [0, 0, 90]
        boxes = rot_x(boxes, random_rots[0])
        boxes = rot_y(boxes, random_rots[1])
        boxes = rot_z(boxes, random_rots[2])
        return boxes, random_rots

    def _augment_pos(self, boxes):
        boxes = boxes.reshape(-1, 6)
        centers = 0.5 * (boxes[:, :3] + boxes[:, 3:])
        xmin, ymin, zmin = centers.min(0).flatten().tolist()
        xmax, ymax, zmax = centers.max(0).flatten().tolist()
        x_tr = np.random.uniform(-xmin - self.lens[0], self.lens[0] - xmax, 1)
        y_tr = np.random.uniform(-ymin - self.lens[1], self.lens[1] - ymax, 1)
        z_tr = np.random.uniform(-zmin - self.lens[2], self.lens[2] - zmax, 1)
        boxes[:, (0, 3)] += x_tr[None, :]
        boxes[:, (1, 4)] += y_tr[None, :]
        boxes[:, (2, 5)] += z_tr[None, :]
        return boxes.reshape(-1, 3)

    def _augment_scale(self, label, boxes):
        if label in (
            'square', 'circle', 'line', 'cube', 'triangle',
            # 'pizza', 'racer', 'alpha', 'caravan'
        ):
            boxes = boxes.reshape(-1, 6)
            imax = np.abs((boxes[:, :2] + boxes[:, 3:5]) / 2).max(0)
            scale = np.random.uniform(
                2 - self.lens[0] / (imax[0] + 1e-5),
                self.lens[1] / (imax[1] + 1e-5),
                1
            )
            boxes = boxes.reshape(-1, 3)
        else:
            imax = np.abs(boxes).max()
            scale = np.random.uniform(0.8, 1 / (imax + 1e-5), 1)
        return boxes * scale

    @staticmethod
    def _augment_size(label, boxes):
        """Boxes should be center-size here."""
        if label in ('square', 'circle', 'line', 'cube', 'triangle'):
            scale = np.random.uniform(0.9, 1.1, len(boxes))
            boxes[:, 3:] = boxes[:, 3:] * scale[:, None]
            boxes[:, 3:] = np.minimum(boxes[:, 3:], SIZE)
            boxes[:, 3:] = np.maximum(boxes[:, 3:], 0.1)
        return boxes


def rot_x(pc, theta):
    """Rotate along x-axis."""
    theta = theta * np.pi / 180
    return np.matmul(
        np.array([
            [1.0, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ]),
        pc.T
    ).T


def rot_y(pc, theta):
    """Rotate along y-axis."""
    theta = theta * np.pi / 180
    return np.matmul(
        np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1.0, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ]),
        pc.T
    ).T


def rot_z(pc, theta):
    """Rotate along z-axis."""
    theta = theta * np.pi / 180
    return np.matmul(
        np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1.0]
        ]),
        pc.T
    ).T
