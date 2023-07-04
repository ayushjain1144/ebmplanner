"""Dataset class for training EBMs on simple concepts."""

from copy import deepcopy
import json
import os.path as osp

import numpy as np
from shapely.geometry import MultiPoint
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
import math

from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.colors import TABLEAU_COLORS
import numpy as np
import io
from PIL import Image
import torch


def _corner_points_2d(box):
    """Convert min-max box to corner points."""
    [xmin, ymin, xmax, ymax] = box
    c1 = np.array([xmin, ymin])
    c2 = np.array([xmax, ymin])
    c3 = np.array([xmax, ymax])
    c4 = np.array([xmin, ymax])
    return np.vstack([c1, c2, c3, c4])


def plot_box_2d(box, axis, c):
    """Plot a box with a given color."""
    box = np.concatenate((box[:2] - box[2:] / 2, box[:2] + box[2:] / 2))
    corners = _corner_points_2d(box)
    axis.plot(
        [corners[0, 0], corners[1, 0]],
        [corners[0, 1], corners[1, 1]],
        c=c
    )
    axis.plot(
        [corners[1, 0], corners[2, 0]],
        [corners[1, 1], corners[2, 1]],
        c=c
    )
    axis.plot(
        [corners[2, 0], corners[3, 0]],
        [corners[2, 1], corners[3, 1]],
        c=c
    )
    axis.plot(
        [corners[3, 0], corners[0, 0]],
        [corners[3, 1], corners[0, 1]],
        c=c
    )

    return axis


def plot_relations_2d(rel_boxes, ref_boxes=None):
    # boxes: [B, N, 2]
    rel_boxes = rel_boxes[0]
    ref_boxes = ref_boxes[0] if ref_boxes is not None else None

    # Initialize figure
    fig = plt.figure(figsize=(12, 12))
    axis = fig.add_subplot(1, 1, 1)
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)
    colors = list(TABLEAU_COLORS.keys())

    for b, box in enumerate(rel_boxes):
        if box.sum() == 0:
            continue
        rect = patches.Rectangle(
            (box[0] - box[2]*0.5, box[1] - box[3]*0.5), box[2], box[3],
            linewidth=2, edgecolor=colors[b % 4], facecolor='none'
        )
        axis.add_patch(rect)
    plt.show()

    if ref_boxes is not None:
        for box in ref_boxes:
            if box.sum() == 0:
                continue
            rect = patches.Rectangle(
                (box[0] - box[2]*0.5, box[1] - box[3]*0.5), box[2], box[3],
                linewidth=2, edgecolor='b', facecolor='none'
            )
            axis.add_patch(rect)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = np.array(Image.open(buf))  # H x W x 4
    image = image[:, :, :3]
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).unsqueeze(0)  # 1 x 3 x H x W
    plt.close('all')
    return image

class EBMDataset(Dataset):
    """Dataset utilities for concept learning."""

    def __init__(self, split='train', concept='right'):
        """Initialize dataset."""
        self.split = split
        self.concept = concept
        self.annos = self.load_annos()
        print('Loaded %d samples for %s' % (len(self.annos), concept))

    def load_annos(self):
        """
        Load annotations.

        {
            'label': label,
            'ref_boxes': ref_boxes.tolist(),  # boxes of reference
            'rel_boxes': boxes.tolist(),  # in relation with reference
            'attention': attention_inds.tolist()  # map to rel_boxes
        }
        """
        split = 'train' if self.split == 'train' else 'test'
        if not osp.exists(f'data/{self.concept}.json'):
            create_graph_data(
                f'data/{self.concept}.json',
                self.concept
            )
        with open(f'data/{self.concept}.json') as fid:
            annos = json.load(fid)[split]
        if self.split == 'train':
            data = np.copy(annos)
            annos = []
            for _ in range(500):
                annos += data.tolist()
        return annos

    def __getitem__(self, index):
        """Get current batch for input index."""
        if self.concept == 'center':
            return self._get_center(index)
        if self.concept in ('square', 'circle', 'line', 'triangle', 'stack'):
            return self._get_shape(index)
        return self._get_relation(index)

    def _get_relation(self, index):
        anno = deepcopy(self.annos[index])
        # Fill missing values
        t = np.array(anno['attention']).argmax()
        rel_boxes = [anno['rel_boxes'][t]]
        # Convert boxes to 'points' (useful for augmentation)
        boxes = np.asarray(anno['ref_boxes'] + rel_boxes + [anno['rel_boxes'][t-1]])
        boxes = np.concatenate((
            boxes[:, :3] - boxes[:, 3:] / 2, boxes[:, :3] + boxes[:, 3:] / 2
        ), 1)
        boxes = boxes.reshape(len(boxes) * 2, 3)
        # Augmentation
        if self.split == 'train':
            boxes = self._augment_rot(anno['label'], boxes)
            boxes = self._augment_pos(boxes)
            boxes = self._augment_scale(anno['label'], boxes)
        # Back to boxes (x_c, y_c, z_c, W, D, H)
        boxes = boxes.reshape(-1, 6)
        centers = (boxes[:, :3] + boxes[:, 3:]) / 2
        size = np.abs(boxes[:, 3:] - boxes[:, :3])
        boxes = np.concatenate((centers[..., :2], size[..., :2]), 1)
        boxes[-1, 2:] = boxes[1, 2:]
        boxes[-1, :2] = np.random.uniform(-0.85, 0.85, 2)
        return {
            "class_name": anno['label'],
            "ref": torch.from_numpy(boxes[:1]).float(),
            "rel": torch.from_numpy(boxes[1:2]).float(),
            "neg": torch.from_numpy(boxes[2:3]).float()
        }

    def _get_center(self, index):
        anno = deepcopy(self.annos[index])
        # Fill missing values
        t = np.array(anno['attention']).argmin()
        # Convert boxes to 'points' (useful for augmentation)
        boxes = np.asarray(anno['ref_boxes'])
        boxes = np.concatenate((
            boxes[:, :3] - boxes[:, 3:] / 2, boxes[:, :3] + boxes[:, 3:] / 2
        ), 1)
        boxes = boxes.reshape(len(boxes) * 2, 3)
        # Augmentation
        if self.split == 'train':
            boxes = self._augment_rot(anno['label'], boxes)
            boxes = self._augment_pos(boxes)
            boxes = self._augment_scale(anno['label'], boxes)
        # Back to boxes (x_c, y_c, z_c, W, D, H)
        boxes = boxes.reshape(-1, 6)
        centers = (boxes[:, :3] + boxes[:, 3:]) / 2
        size = np.abs(boxes[:, 3:] - boxes[:, :3])
        boxes = np.concatenate((centers[..., :2], size[..., :2]), 1)
        # Handle center
        boxes[t, :2] = np.concatenate((boxes[:t], boxes[t+1:]))[:, :2].mean(0)
        neg = np.random.uniform(-0.85, 0.85, 4)
        neg[2:] = boxes[t, 2:]
        return {
            "class_name": anno['label'],
            "ref": torch.from_numpy(np.concatenate((boxes[:t], boxes[t+1:]))).float(),
            "rel": torch.from_numpy(boxes[t]).float().unsqueeze(0),
            "neg": torch.from_numpy(neg).float().unsqueeze(0)
        }

    def _get_shape(self, index):
        anno = deepcopy(self.annos[index])
        # Fill missing values
        t = np.array(anno['attention'])
        boxes = np.array(anno['ref_boxes'])[t > 0]
        # Convert boxes to 'points' (useful for augmentation)
        boxes = np.concatenate((
            boxes[:, :3] - boxes[:, 3:] / 2, boxes[:, :3] + boxes[:, 3:] / 2
        ), 1)
        boxes = boxes.reshape(len(boxes) * 2, 3)
        # Augmentation
        if self.split == 'train':
            boxes = self._augment_rot(anno['label'], boxes)
            boxes = self._augment_pos(boxes)
            boxes = self._augment_scale(anno['label'], boxes)
        # Back to boxes (x_c, y_c, z_c, W, D, H)
        boxes = boxes.reshape(-1, 6)
        centers = (boxes[:, :3] + boxes[:, 3:]) / 2
        size = np.abs(boxes[:, 3:] - boxes[:, :3])
        boxes = np.concatenate((centers[..., :2], size[..., :2]), 1)
        neg_boxes = np.copy(boxes)
        neg_boxes[:, :2] = np.random.uniform(-0.85, 0.85, (len(neg_boxes), 2))
        # Center and length as ref
        centers = boxes[:, :2].mean(0)
        lengths = 2 * ((boxes[:, :2] - centers[None]) ** 2).sum(1).max()
        ref_boxes = np.array(centers.tolist() + [lengths])
        return {
            "class_name": anno['label'],
            "ref": torch.from_numpy(ref_boxes).float(),
            "rel": torch.from_numpy(boxes).float(),
            "neg": torch.from_numpy(neg_boxes).float(),
            "attention": torch.ones(len(boxes))
        }

    def __len__(self):
        """Return number of samples."""
        return len(self.annos)

    @staticmethod
    def _augment_rot(label, boxes):
        if label in ('square', 'circle', 'line', 'cube', 'triangle', 'stack'):
            max_degrees = (0, 0, 180)
        elif label in ('left', 'right'):
            max_degrees = (180, 3, 3)
        elif label in ('front', 'behind'):
            max_degrees = (3, 180, 3)
        elif label in ('above', 'below', 'supported-by', 'supporting'):
            max_degrees = (3, 3, 180)
        else:
            max_degrees = (180, 180, 180)
        boxes = rot_x(boxes, (2 * np.random.rand() - 1) * max_degrees[0])
        boxes = rot_y(boxes, (2 * np.random.rand() - 1) * max_degrees[1])
        boxes = rot_z(boxes, (2 * np.random.rand() - 1) * max_degrees[2])
        return boxes

    @staticmethod
    def _augment_pos(boxes):
        boxes = boxes.reshape(-1, 6)
        centers = 0.5 * (boxes[:, :3] + boxes[:, 3:])
        xmin, ymin, zmin = centers.min(0).flatten().tolist()
        xmax, ymax, zmax = centers.max(0).flatten().tolist()
        x_translation = np.random.uniform(-xmin - 0.85, 0.85 - xmax, 1)
        y_translation = np.random.uniform(-ymin - 0.85, 0.85 - ymax, 1)
        z_translation = np.random.uniform(-zmin - 0.85, 0.85 - zmax, 1)
        boxes[:, (0, 3)] += x_translation[None, :]
        boxes[:, (1, 4)] += y_translation[None, :]
        boxes[:, (2, 5)] += z_translation[None, :]
        return boxes.reshape(-1, 3)

    @staticmethod
    def _augment_scale(label, boxes):
        if label in ('square', 'circle', 'line', 'cube', 'triangle'):
            boxes = boxes.reshape(-1, 6)
            imax = np.abs((boxes[:, :2] + boxes[:, 3:5]) / 2).max()
            scale = np.random.uniform(
                2 - 0.85 / (imax + 1e-5), 0.85 / (imax + 1e-5), 1
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
            boxes[:, 3:] = np.minimum(boxes[:, 3:], 0.3)
            boxes[:, 3:] = np.maximum(boxes[:, 3:], 0.1)
        return boxes


def ebm_collate_fn(batch):
    """Collate function for concept EBMs."""
    return_dict = {
        "class_names": [ex["class_name"] for ex in batch],
        "ref_boxes": pad_sequence(
            [ex["ref"] for ex in batch],
            batch_first=True, padding_value=0
        ),
        "rel_boxes": pad_sequence(
            [ex["rel"] for ex in batch],
            batch_first=True, padding_value=0
        ),
        "neg_boxes": pad_sequence(
            [ex["neg"] for ex in batch],
            batch_first=True, padding_value=0
        )
    }
    if 'attention' in batch[0]:
        return_dict["attentions"] = pad_sequence(
            [ex["attention"] for ex in batch],
            batch_first=True, padding_value=0
        )
    return return_dict


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


class EBMDataGenerator:
    """A class to construct positive samples for a concept EBM."""

    def __init__(self):
        """Specify the number of samples to create per class."""
        self.executors = {
            'left': self._is_left,
            'right': self._is_right,
            'front': self._is_front,
            'behind': self._is_behind,
            'supported-by': self._is_supported_by,
            'supporting': self._is_supporting,
            'above': self._is_above,
            'below': self._is_below,
            'higher-than': self._is_higher,
            'lower-than': self._is_lower,
            'equal-height': self._is_equal_height,
            'larger-than': self._is_larger,
            'smaller-than': self._is_smaller,
            'equal_size': self._is_equal_size,
            'largest': self._get_largest,
            'smallest': self._get_smallest,
            'closest': self._get_closest,
            'furthest': self._get_furthest,
            'between': self._is_between,
            'line': self.sample_nboxes_line,
            'circle': self.sample_nboxes_circle,
            'center': self.sample_nboxes_circle,
            'square': self.sample_nboxes_square,
            'triangle': self.sample_nboxes_triangle,
            'cube': self.sample_nboxes_cube,
            'inside': self.sample_nboxes_inside,
            'stack': self.sample_nboxes_stack
        }

    def sample(self, nsamples, labels=None, num = 10):
        """Sample a given number of positives for each class."""
        if not labels:
            labels = self.executors.keys()
        samples = []
        for label in labels:
            print('Data for ' + label)
            label_samples = [
                self.sample_one(label,num) for _ in tqdm(range(nsamples))
            ]
            samples += list(label_samples)
        return samples

    def sample_one(self, label, num = 10):
        """Sample a positive for a given label."""
        # Sample reference boxes
        if label == 'between':
            return self._sample_two_args(label)
        if label in ('largest', 'smallest'):
            return self._sample_size_comp(label)
        if label in ('closest', 'furthest'):
            return self._sample_proximal(label)
        if label in ('line', 'stack'):
            return self._sample_shape(
                label, np.random.randint(2, 7), 0
            )
        if label == 'circle':
            return self._sample_shape(
                # label, np.random.randint(6, 12), 0
                label, num, 0
            )
        if label == 'center':
            return self._sample_shape(
                label, np.random.randint(3, 12), 1
            )
        if label == 'square':
            return self._sample_shape(label, 4, 0)
        if label == 'triangle':
            return self._sample_shape(label, 3, 0)
        if label == 'cube':
            return self._sample_shape(label, 8, np.random.randint(2, 6))
        if label == 'inside':
            return self._sample_inside(label)
        if label in ('supporting', 'supported-by'):
            return self._sample_support(label)
        return self._sample_one_arg(label)

    def _sample_support(self, label):
        # Get a pair that satisfies the relation
        found = False
        while not found:
            ref_boxes = self.sample_nboxes(2)
            if label == 'supporting':
                ref_boxes[0, 2] = max(
                    ref_boxes[0, 2],
                    ref_boxes[0, 5] + ref_boxes[1, 5] - 1
                )
                ref_boxes[1, 2] = (
                    ref_boxes[0, 2]
                    - (ref_boxes[0, 5] + ref_boxes[1, 5]) / 2
                    + 0.02 * np.random.rand(1)[0] - 0.01
                )
            else:
                ref_boxes[1, 2] = max(
                    ref_boxes[1, 2],
                    ref_boxes[1, 5] + ref_boxes[0, 5] - 1
                )
                ref_boxes[0, 2] = (
                    ref_boxes[1, 2]
                    - (ref_boxes[1, 5] + ref_boxes[0, 5]) / 2
                    + 0.02 * np.random.rand(1)[0] - 0.01
                )
            mmboxes = self._to_min_max(ref_boxes)
            if self.executors[label](mmboxes[1], mmboxes[0]):
                found = True
        # Sample negatives
        n_boxes = np.random.randint(2, 7)  # how many boxes to sample
        target = np.random.randint(n_boxes)
        attention_inds = np.zeros(n_boxes)
        attention_inds[target] = 1
        boxes = [np.asarray(box) for box in ref_boxes]
        ref_boxes = ref_boxes[:1]
        rf_bxs = self._to_min_max(ref_boxes)
        ind = 0
        while len(boxes) < n_boxes + 2:
            if ind == target:
                boxes.append(np.copy(boxes[1]))
                ind += 1
            else:
                new_box = self.sample_nboxes(1, boxes)
                mmbox = self._to_min_max(new_box).flatten()
                if not self.executors[label](mmbox, rf_bxs[0]):
                    boxes.append(np.copy(new_box.flatten()))
                    ind += 1
        boxes = np.stack(boxes[len(ref_boxes) + 1:])
        return {
            'label': label,
            'ref_boxes': ref_boxes.tolist(),
            'rel_boxes': boxes.tolist(),
            'attention': attention_inds.tolist()
        }

    def _sample_one_arg(self, label):
        # Get a pair that satisfies the relation
        found = False
        while not found:
            ref_boxes = self.sample_nboxes(2)
            mmboxes = self._to_min_max(ref_boxes)
            if self.executors[label](mmboxes[0], mmboxes[1]):
                found = True
                ref_boxes = ref_boxes[::-1]
            elif self.executors[label](mmboxes[1], mmboxes[0]):
                found = True
        # Sample negatives
        n_boxes = np.random.randint(2, 7)  # how many boxes to sample
        target = np.random.randint(n_boxes)
        attention_inds = np.zeros(n_boxes)
        attention_inds[target] = 1
        boxes = [np.asarray(box) for box in ref_boxes]
        ref_boxes = ref_boxes[:1]
        rf_bxs = self._to_min_max(ref_boxes)
        ind = 0
        while len(boxes) < n_boxes + 2:
            if ind == target:
                boxes.append(np.copy(boxes[1]))
                ind += 1
            else:
                new_box = self.sample_nboxes(1, boxes)
                mmbox = self._to_min_max(new_box).flatten()
                if not self.executors[label](mmbox, rf_bxs[0]):
                    boxes.append(np.copy(new_box.flatten()))
                    ind += 1
        boxes = np.stack(boxes[len(ref_boxes) + 1:])
        return {
            'label': label,
            'ref_boxes': ref_boxes.tolist(),
            'rel_boxes': boxes.tolist(),
            'attention': attention_inds.tolist()
        }

    def _sample_two_args(self, label):
        # Get a pair that satisfies the relation
        found = False
        while not found:
            ref_boxes = self.sample_nboxes(3)
            mmboxes = self._to_min_max(ref_boxes)
            if self.executors[label](mmboxes[0], mmboxes[1], mmboxes[2]):
                found = True
                ref_boxes = ref_boxes[(1, 2, 0), :]
            elif self.executors[label](mmboxes[1], mmboxes[0], mmboxes[2]):
                found = True
                ref_boxes = ref_boxes[(0, 2, 1), :]
            elif self.executors[label](mmboxes[2], mmboxes[0], mmboxes[1]):
                found = True
        # Sample negatives
        n_boxes = np.random.randint(2, 7)  # how many boxes to sample
        target = np.random.randint(n_boxes)
        attention_inds = np.zeros(n_boxes)
        attention_inds[target] = 1
        boxes = [np.asarray(box) for box in ref_boxes]
        ref_boxes = ref_boxes[:2]
        rf_bxs = self._to_min_max(ref_boxes)
        ind = 0
        while len(boxes) < n_boxes + 3:
            if ind == target:
                boxes.append(np.copy(boxes[2]))
                ind += 1
            else:
                new_box = self.sample_nboxes(1, boxes)
                mmbox = self._to_min_max(new_box).flatten()
                if not self.executors[label](mmbox, rf_bxs[0], rf_bxs[1]):
                    boxes.append(np.copy(new_box.flatten()))
                    ind += 1
        boxes = np.stack(boxes[len(ref_boxes) + 1:])
        return {
            'label': label,
            'ref_boxes': ref_boxes.tolist(),
            'rel_boxes': boxes.tolist(),
            'attention': attention_inds.tolist()
        }

    def _sample_proximal(self, label):
        ref_boxes = self.sample_nboxes(1)  # reference box
        n_boxes = 7  # how many boxes to sample
        boxes = self.sample_nboxes(n_boxes, ref_boxes)
        attention_inds = np.zeros(n_boxes)
        ind = self.executors[label](
            self._to_min_max(boxes),
            self._to_min_max(ref_boxes)[0]
        )  # index of closest/furthest
        attention_inds[ind] = 1
        return {
            'label': label,
            'ref_boxes': ref_boxes.tolist(),
            'rel_boxes': boxes.tolist(),
            'attention': attention_inds.tolist()
        }

    def _sample_size_comp(self, label):
        n_boxes = 7  # how many boxes to sample
        boxes = self.sample_nboxes(n_boxes)
        attention_inds = np.zeros(n_boxes)
        ind = self.executors[label](self._to_min_max(boxes))
        attention_inds[ind] = 1
        return {
            'label': label,
            'ref_boxes': [[0.0, 0, 0, 0, 0, 0]],
            'rel_boxes': boxes.tolist(),
            'attention': attention_inds.tolist()
        }

    def _sample_shape(self, label, num_boxes, num_dis_boxes):
        # Sample boxes that form the shape
        all_ref_boxes = self.executors[label](num_boxes)
        # Sample distractor boxes
        # constrain_z = np.unique(ref_boxes[:, 2]).tolist()
        # dis_boxes = self.sample_nboxes(num_dis_boxes, ref_boxes, constrain_z)
        # Concatenate
        # all_ref_boxes = np.concatenate((ref_boxes, dis_boxes), axis=0)
        # Attention
        attention = np.ones(num_boxes + num_dis_boxes)
        # attention[:num_boxes] = 1
        # Shuffle
        # shuffler = np.random.permutation(num_boxes + num_dis_boxes)
        # all_ref_boxes = all_ref_boxes[shuffler]
        # attention = attention[shuffler]
        # Return
        return {
            'label': label,
            'ref_boxes': all_ref_boxes.tolist(),
            'rel_boxes': None,  # no rel boxes
            'attention': attention.tolist()
        }

    def _sample_inside(self, label):
        # Sample reference box
        ref_boxes = self.sample_nboxes(1)
        # Sample box that satisfies the concept
        boxes = self.sample_nboxes_inside(1, ref_boxes)
        # Sample distractor boxes
        num_dis_boxes = np.random.randint(2, 6)
        dis_boxes = self.sample_nboxes(num_dis_boxes, ref_boxes)
        # Concatenate
        all_rel_boxes = np.concatenate((boxes, dis_boxes), axis=0)
        # Attention
        attention = np.zeros(len(all_rel_boxes))
        attention[0] = 1
        # Shuffle
        shuffler = np.random.permutation(len(attention))
        all_rel_boxes = all_rel_boxes[shuffler]
        attention = attention[shuffler]
        # Return
        return {
            'label': label,
            'ref_boxes': ref_boxes.tolist(),
            'rel_boxes': all_rel_boxes.tolist(),
            'attention': attention.tolist()
        }

    def sample_nboxes(self, n_points, old_boxes=None, constrain_z=[]):
        """Sample n boxes in the space (cube [-0.85, 0.85])."""
        boxes = []
        if old_boxes is not None:
            boxes += [np.asarray(box) for box in old_boxes]
        else:
            old_boxes = []
        while len(boxes) < n_points + len(old_boxes):
            new_box = np.concatenate((
                1.7 * np.random.rand(3) - 0.85,  # center
                0.2 * np.random.rand(3) + 0.1  # size
            ))
            if constrain_z:
                new_box[2] = constrain_z[np.random.randint(len(constrain_z))]
            if not any(self._intersect(new_box, box) for box in boxes):
                boxes.append(np.copy(new_box))
        return np.stack(boxes)[-n_points:]

    def sample_nboxes_line(self, n_points):
        """Sample n boxes in the space forming a line."""
        # Define a line
        x1, y1, z1, x2, y2 = 1.7 * np.random.rand(5) - 0.85
        z1 = 0
        line = [(x1, y1, z1), (x2, y2, z1)]
        # Sample boxes
        boxes = []
        while len(boxes) < n_points:
            # Random box
            new_box = np.concatenate((
                1.7 * np.random.rand(3) - 0.85,  # center
                0.1 * np.random.rand(3) + 0.1  # size
            ))
            new_box[0] = len(boxes) * (x2 - x1) / n_points + x1
            # Condition random y, z to x to satify line concept
            new_box[1], new_box[2] = self.get_yz_on_line(new_box[0], line)
            # Check if the new box intersects an existing
            if not any(self._intersect(new_box, box) for box in boxes):
                boxes.append(np.copy(new_box))
        return np.stack(boxes)

    @staticmethod
    def get_yz_on_line(x, line):
        """Given x and line [(x1, y1, z1), (x2, y2, z2)], find y, z."""
        x1, y1, z1 = line[0]
        x2, y2, z2 = line[1]
        y = (((y2-y1) / (x2-x1)) * (x - x1)) + y1
        z = (((z2-z1) / (x2-x1)) * (x - x1)) + z1
        return y, z

    def sample_nboxes_circle(self, n_points):
        """Sample n boxes in the space forming a circle."""
        # Sample center and radius
        while True:
            # center
            xc, yc = 1.7 * np.random.rand(2) - 0.85

            # circle should fit in our space
            low = 0.3
            high = 0.85 - max(abs(xc), abs(yc))

            if high < low:
                continue

            radius = np.random.uniform(low=low, high=high)
            break
        zc = 0
        # Circle
        # circle = [xc, yc, radius]

        # Sample boxes
        boxes = []
        for i in range(n_points):
            add_it = False
            while not add_it:
                # Random box
                new_box = np.concatenate((
                    1.7 * np.random.rand(3) - 0.85,  # center
                    0.5 * radius * (0.2 * np.random.rand(3) + 0.1)  # size
                ))
                # Resample x within limits
                new_box[0] = xc + np.cos(2 * np.pi * i / n_points) * radius
                new_box[1] = yc + np.sin(2 * np.pi * i / n_points) * radius
                # Condition random y to x to satify circle concept
                # new_box[1] = self.get_y_on_circle(new_box[0], circle)
                new_box[2] = zc
                # Check if the new box intersects an existing
                add_it = not any(self._intersect(new_box, box) for box in boxes)
            boxes.append(np.copy(new_box))
        return np.stack(boxes)

    @staticmethod
    def get_y_on_circle(x, circle):
        """Given x and circle [xc, yc, radius], find y."""
        xc, yc, radius = circle
        sign = np.random.choice([-1, 1])
        y = sign * (math.sqrt(radius**2 - (x - xc) ** 2)) + yc
        return y

    def sample_nboxes_square(self, n_points):
        """Sample n boxes in the space forming a square."""
        # Sample center and length
        while True:
            # sample center
            xc, yc = 1.7 * np.random.rand(2) - 0.85

            # circle should fit in our space
            low = 0.3
            high = 0.85 - max(abs(xc), abs(yc))

            if high < low:
                continue

            length = np.random.uniform(low=low, high=high)
            break
        zc = 1.7 * np.random.rand(1)[0] - 0.85
        pos = [
            (xc-length, yc-length), (xc-length, yc+length),
            (xc+length, yc-length), (xc+length, yc+length)
        ]

        # Sample boxes
        boxes = []
        while len(boxes) < n_points:
            # Random box
            new_box = np.concatenate((
                1.7 * np.random.rand(3) - 0.85,  # center
                0.2 * np.random.rand(3) + 0.1  # size
            ))
            # Condition x, y to square
            new_box[0], new_box[1] = pos[len(boxes)]
            new_box[2] = zc
            # Check if the new box intersects an existing
            if not any(self._intersect(new_box, box) for box in boxes):
                boxes.append(np.copy(new_box))
        return np.stack(boxes)

    def sample_nboxes_triangle(self, n_points):
        """Sample n boxes in the space forming a triangle."""
        return self.sample_nboxes_circle(3)
        # Sample base of the triangle
        x1, y1, x2, y2 = 1.7 * np.random.rand(4) - 0.85
        X, Y = np.array([x1, x2]), np.array([y1, y2])
        m = (X + Y) / 2
        o = (X - m) * 3**0.5
        t = np.array([[0, -1], [1, 0]])
        x3, y3 = m + o @ t
        pos = [(x1, y1), (x2, y2), (x3, y3)]
        z = 1.7 * np.random.rand(1)[0] - 0.85

        # Sample boxes
        boxes = []
        while len(boxes) < n_points:
            # Random box
            new_box = np.concatenate((
                1.7 * np.random.rand(3) - 0.85,  # center
                0.2 * np.random.rand(3) + 0.1  # size
            ))
            # Condition x, y on triangle
            ind = len(boxes)
            new_box[0], new_box[1] = pos[ind]
            new_box[2] = z
            # Check if the new box intersects an existing
            if not any(self._intersect(new_box, box) for box in boxes):
                boxes.append(np.copy(new_box))
        return np.stack(boxes)

    def sample_nboxes_cube(self, n_points):
        """Sample n boxes in the space forming a cube."""
        # Sample center, range of a cude that can fit
        while True:
            # sample center
            xc, yc, zc = 1.7 * np.random.rand(3) - 0.85

            # length range
            low = 0.3
            high = 0.85 - max(abs(xc), abs(yc), abs(zc))

            if high < low:
                continue
            length = np.random.uniform(low=low, high=high)
            break
        pos = [
            (xc-length/2.0, yc-length/2.0, zc-length/2.0),
            (xc-length/2.0, yc-length/2.0, zc+length/2.0),
            (xc-length/2.0, yc+length/2.0, zc-length/2.0),
            (xc+length/2.0, yc-length/2.0, zc-length/2.0),
            (xc+length/2.0, yc+length/2.0, zc-length/2.0),
            (xc+length/2.0, yc-length/2.0, zc+length/2.0),
            (xc-length/2.0, yc+length/2.0, zc+length/2.0),
            (xc+length/2.0, yc+length/2.0, zc+length/2.0)
        ]

        # Sample boxes
        boxes = []
        while len(boxes) < n_points:
            # Random box
            new_box = np.concatenate((
                1.7 * np.random.rand(3) - 0.85,  # center
                0.2 * np.random.rand(3) + 0.1  # size
            ))
            # Condition x, y, z on cube
            ind = len(boxes)
            new_box[0], new_box[1], new_box[2] = pos[ind]
            # Check if the new box intersects an existing
            if not any(self._intersect(new_box, box) for box in boxes):
                boxes.append(np.copy(new_box))
        return np.stack(boxes)

    def sample_nboxes_inside(self, n_points, old_boxes=None):
        """Sample n boxes in the space (cube [-0.85, 0.85])."""
        boxes = []
        if old_boxes is not None:
            boxes += [np.asarray(box) for box in old_boxes]
        else:
            assert(False)

        # currently we support n_points == 1
        assert(n_points == 1)

        x_old, y_old, z_old = old_boxes[0][:3]
        l_old, w_old, h_old = old_boxes[0][3:]
        l_new = np.random.uniform(low=0.1, high=l_old)
        w_new = np.random.uniform(low=0.1, high=w_old)
        h_new = np.random.uniform(low=0.1, high=h_old)

        x_max = x_old + (l_old - l_new)
        x_min = x_old - (l_old - l_new)
        y_max = y_old + (w_old - w_new)
        y_min = y_old - (w_old - w_new)
        z_max = z_old + (h_old - h_new)
        z_min = z_old - (h_old - h_new)

        while True:
            x_new = np.random.uniform(low=x_min, high=x_max)
            y_new = np.random.uniform(low=y_min, high=y_max)
            z_new = np.random.uniform(low=z_min, high=z_max)
            new_box = np.array([x_new, y_new, z_new, l_new, w_new, h_new])

            if any(self._inside(new_box, box) for box in boxes):
                boxes.append(np.copy(new_box))
                break
        return np.stack(boxes)[-n_points:]

    def sample_nboxes_stack(self, n_points):
        """Sample n boxes in the space forming a stack."""
        boxes = []

        # generate all boxes first
        height_from_ground = -0.5

        while len(boxes) < n_points:

            new_box = np.concatenate((
                1.7 * np.random.rand(3) - 0.85,  # center
                0.2 * np.random.rand(3) + 0.1  # size
            ))
            # st()
            new_box[2] = height_from_ground + (new_box[5] / 2.0)
            new_box[0] = 0
            new_box[1] = 0
            boxes.append(np.copy(new_box))
            height_from_ground += new_box[5]

        return np.stack(boxes)

    @staticmethod
    def box2points(box):
        """Convert box min/max coordinates to vertices (8x3)."""
        x_min, y_min, z_min, x_max, y_max, z_max = box
        return np.array([
            [x_min, y_min, z_min], [x_min, y_max, z_min],
            [x_max, y_min, z_min], [x_max, y_max, z_min],
            [x_min, y_min, z_max], [x_min, y_max, z_max],
            [x_max, y_min, z_max], [x_max, y_max, z_max]
        ])

    @staticmethod
    def _compute_dist(points0, points1):
        """Compute minimum distance between two sets of points."""
        dists = ((points0[:, None, :] - points1[None, :, :]) ** 2).sum(2)
        return dists.min()

    def _intersect(self, box_a, box_b):
        return self._intersection_vol(box_a, box_b) > 0

    @staticmethod
    def _intersection_vol(box_a, box_b):
        xA = max(box_a[0] - box_a[3] / 2, box_b[0] - box_b[3] / 2)
        yA = max(box_a[1] - box_a[4] / 2, box_b[1] - box_b[4] / 2)
        zA = max(box_a[2] - box_a[5] / 2, box_b[2] - box_b[5] / 2)
        xB = min(box_a[0] + box_a[3] / 2, box_b[0] + box_b[3] / 2)
        yB = min(box_a[1] + box_a[4] / 2, box_b[1] + box_b[4] / 2)
        zB = min(box_a[2] + box_a[5] / 2, box_b[2] + box_b[5] / 2)
        return max(0, xB - xA) * max(0, yB - yA) * max(0, zB - zA)

    def _inside(self, box_a, box_b):
        volume_a = box_a[3] * box_a[4] * box_a[5]
        return np.isclose(self._intersection_vol(box_a, box_b), volume_a)

    @staticmethod
    def iou_2d(box0, box1):
        """Compute 2d IoU for two boxes in coordinate format."""
        box_a = np.array(box0)[(0, 1, 3, 4), ]
        box_b = np.array(box1)[(0, 1, 3, 4), ]
        # Intersection
        xA = max(box_a[0], box_b[0])
        yA = max(box_a[1], box_b[1])
        xB = min(box_a[2], box_b[2])
        yB = min(box_a[3], box_b[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        # Areas
        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        # Return IoU and area ratios
        return (
            inter_area / (box_a_area + box_b_area - inter_area),  # iou
            [inter_area / box_a_area, inter_area / box_b_area],
            [box_a_area / box_b_area, box_b_area / box_a_area]
        )

    @staticmethod
    def volume(box):
        """Compute of box in coordinate format (min, max)."""
        return (box[3] - box[0]) * (box[4] - box[1]) * (box[5] - box[2])

    @staticmethod
    def _to_min_max(box):
        return np.concatenate((
            box[:, :3] - box[:, 3:] / 2, box[:, :3] + box[:, 3:] / 2
        ), 1)

    @staticmethod
    def _same_x_range(box, ref_box):
        return (
            min(box[3], ref_box[3]) - max(box[0], ref_box[0])
            > 0.8 * min(box[3] - ref_box[0], box[3] - ref_box[0])
        )

    @staticmethod
    def _same_y_range(box, ref_box):
        return (
            min(box[4], ref_box[4]) - max(box[1], ref_box[1])
            > 0.8 * min(box[4] - ref_box[1], box[4] - ref_box[1])
        )

    @staticmethod
    def _same_z_range(box, ref_box):
        return (
            min(box[5], ref_box[5]) - max(box[2], ref_box[2])
            > 0.2 * (box[5] - box[2])
        )

    def _is_left(self, box, ref_box):
        return (
            box[3] < ref_box[0]  # x_max < x_ref_min
            and self._same_y_range(box, ref_box)
            and self._same_z_range(box, ref_box)
        )

    def _is_right(self, box, ref_box):
        return (
            box[0] > ref_box[3]  # x_min > x_ref_max
            and self._same_y_range(box, ref_box)
            and self._same_z_range(box, ref_box)
        )

    def _is_front(self, box, ref_box):
        return (
            box[4] < ref_box[1]  # y_max < y_ref_min
            and self._same_x_range(box, ref_box)
            and self._same_z_range(box, ref_box)
        )

    def _is_behind(self, box, ref_box):
        return (
            box[1] > ref_box[4]  # y_min > y_ref_max
            and self._same_x_range(box, ref_box)
            and self._same_z_range(box, ref_box)
        )

    def _is_between(self, box, ref_box0, ref_box1):
        # Get the convex hull of all points of the two anchors
        convex_hull = MultiPoint(
            tuple(map(tuple, self.box2points(ref_box0)[:4, :2]))
            + tuple(map(tuple, self.box2points(ref_box1)[:4, :2]))
        ).convex_hull
        # Get box as polygons
        polygon_t = MultiPoint(
            tuple(map(tuple, self.box2points(box)[:4, :2]))
        ).convex_hull
        # Candidate should fall in the convex_hull polygon
        return (
            convex_hull.intersection(polygon_t).area / polygon_t.area > 0.51
            and self._same_z_range(box, ref_box0)
            and self._same_z_range(box, ref_box1)
        )

    def _is_supported_by(self, box, ref_box):
        box_bottom_ref_top_dist = box[2] - ref_box[5]
        _, intersect_ratios, area_ratios = self.iou_2d(box, ref_box)
        int2box_ratio, _ = intersect_ratios
        box2ref_ratio, _ = area_ratios
        return (
            int2box_ratio > 0.3  # xy intersection
            and abs(box_bottom_ref_top_dist) <= 0.01  # close to surface
            and box2ref_ratio < 1.5  # supporter is usually larger
        )

    def _is_supporting(self, box, ref_box):
        ref_bottom_cox_top_dist = ref_box[2] - box[5]
        _, intersect_ratios, area_ratios = self.iou_2d(box, ref_box)
        _, int2ref_ratio = intersect_ratios
        _, ref2box_ratio = area_ratios
        return (
            int2ref_ratio > 0.3  # xy intersection
            and abs(ref_bottom_cox_top_dist) <= 0.01  # close to surface
            and ref2box_ratio < 1.5  # supporter is usually larger
        )

    def _is_above(self, box, ref_box):
        box_bottom_ref_top_dist = box[2] - ref_box[5]
        _, intersect_ratios, _ = self.iou_2d(box, ref_box)
        int2box_ratio, int2ref_ratio = intersect_ratios
        return (
            box_bottom_ref_top_dist > 0.03  # should be above
            and max(int2box_ratio, int2ref_ratio) > 0.2  # xy intersection
        )

    def _is_below(self, box, ref_box):
        ref_bottom_cox_top_dist = ref_box[2] - box[5]
        _, intersect_ratios, _ = self.iou_2d(box, ref_box)
        int2box_ratio, int2ref_ratio = intersect_ratios
        return (
            ref_bottom_cox_top_dist > 0.03  # should be above
            and max(int2box_ratio, int2ref_ratio) > 0.2  # xy intersection
        )

    @staticmethod
    def _is_higher(box, ref_box):
        return box[2] - ref_box[5] > 0.03

    @staticmethod
    def _is_lower(box, ref_box):
        return ref_box[2] - box[5] > 0.03

    def _is_equal_height(self, box, ref_box):
        return self._same_z_range(box, ref_box)

    def _is_larger(self, box, ref_box):
        return self.volume(box) > 1.1 * self.volume(ref_box)

    def _is_smaller(self, box, ref_box):
        return self.volume(ref_box) > 1.1 * self.volume(box)

    def _is_equal_size(self, box, ref_box):
        return (
            not self._is_larger(box, ref_box)
            and not self._is_smaller(box, ref_box)
            and 0.9 < (box[3] - box[0]) / (ref_box[3] - ref_box[0]) < 1.1
            and 0.9 < (box[4] - box[1]) / (ref_box[4] - ref_box[1]) < 1.1
            and 0.9 < (box[5] - box[2]) / (ref_box[5] - ref_box[2]) < 1.1
        )

    def _get_closest(self, boxes, ref_box):
        dists = np.array([
            self._compute_dist(self.box2points(box), self.box2points(ref_box))
            for box in boxes
        ])
        return dists.argmin()

    def _get_furthest(self, boxes, ref_box):
        dists = np.array([
            self._compute_dist(self.box2points(box), self.box2points(ref_box))
            for box in boxes
        ])
        return dists.argmax()

    def _get_largest(self, boxes, ref_box=None):
        return np.array([self.volume(box) for box in boxes]).argmax()

    def _get_smallest(self, boxes, ref_box=None):
        return np.array([self.volume(box) for box in boxes]).argmin()


def create_graph_data(filename, concept='right'):
    """Create graph annotations."""
    nsamples = 40
    ntrain = 20
    generator = EBMDataGenerator()
    samples = generator.sample(nsamples, [concept])
    assert not len(samples) % nsamples
    nlabels = int(len(samples) / nsamples)
    train_samples = []
    test_samples = []
    for k in range(nlabels):
        train_samples += samples[k * nsamples:k * nsamples + ntrain]
        test_samples += samples[k * nsamples + ntrain:(k + 1) * nsamples]
    with open(filename, 'w') as fid:
        json.dump({'train': train_samples, 'test': test_samples}, fid)


if __name__ == "__main__":
    # create_graph_data("tmp.json", 'center')
    generator = EBMDataGenerator()
    samples = generator.sample(1, ['circle'])
    print(samples)
    print(len(samples[0]['ref_boxes']))
    # from utils.visualizations import plot_relations_2d
    samples = np.array([samples[0]['ref_boxes']])[..., (0, 1, 3, 4)]
    import ipdb;ipdb.set_trace()
    plot_relations_2d(samples)
    dset = EBMDataset(concept='circle')
    for i in range(len(dset)):
        dset.__getitem__(i)
