"""Image dataset."""

import os
import pickle
import warnings

import cv2
import numpy as np
from torch.utils.data import Dataset

from global_vars import BOUNDS, CAMERA_CONFIG, PIXEL_SIZE
import utils.transporter_utils as utils

import faulthandler
faulthandler.enable()


MAX_BOXES = 10


class RavensDataset(Dataset):
    """
    Observation-action-goal dataset.

    Args:
        * path (str): path to annotations.
            Structure under path:
            - action: X.pkl contains a list of dicts like:
                {
                    'pose0': ((3,) array, (4,) array),
                    'pose1': ((3,) array, (4,) array),
                    'pick' : (x, y),
                    'place': (x, y)
                }
                the length of this list is num_actions+1, last element is None
            - color: X.pkl contains a list of (3, 480, 640, 3) tensors
            - depth: X.pkl contains a list of (3, 480, 640) tensors
            - info: X.pkl contains a list of dicts like:
                {
                    'fixed': {id: (3 tuples)}
                        fixed are the objects involved in the task
                        but should not be moved (haha! you thought this def
                        would work for all benchmarks? really?)
                    'move': same format as fixed
                        objects involved in task and should be moved
                    'lang_goal': (str) task description
                    'names': {id: str (name) for all objects}
                }
            - reward: X.pkl contains a list of rewards,
                which is the reward the previous step adds,
                e.g. [0, 0.2, 0.2, 0.2, 0.2, 0.2]
            - vis: each image shows the 3 views for each observation step
        * task_list (list): list of tasks names (str)
        * split (str): 'train', 'val' or 'test'
        * n_demos (int): number of demos per tasks
        * augment (bool): whether to enable augmentations
        * repeat (int): repeat annotations to support larger batch sizes
    """

    def __init__(self, path, task_list=[], split='train', n_demos=0,
                 augment=True, repeat=8, square_pad=False, reshape=None,
                 analogical=False, theta_sigma=60, cliport=False):
        """A simple RGB-D image dataset."""
        self._path = path
        self.task_list = task_list
        self.split = split
        self.augment = (split == 'train') and augment
        repeat = repeat if split == 'train' else 1
        self.square_pad = square_pad
        self.reshape = reshape
        self.analogical = analogical
        self.theta_sigma = theta_sigma
        self.seeds_per_task = {}
        self.load_annos(n_demos, split, repeat)
        self.cliport = cliport

    def load_annos(self, n_demos, split, repeat):
        filenames = []
        seeds = []
        task_names = []
        for task in self.task_list:
            _path = os.path.join(self._path, task + '-' + split, 'action')
            _fnames = sorted(os.listdir(_path))
            _keep = min(n_demos, len(_fnames))
            print(f'Found {len(_fnames)} demos for {task}, keeping {_keep}')
            _fnames = _fnames[:n_demos]
            filenames += _fnames
            seeds += [int(name[(name.find('-') + 1):-4]) for name in _fnames]
            task_names += [task for _ in range(len(_fnames))]

        self.cache = {}
        self.seeds_per_task = {task: [] for task in self.task_list}
        self.seeds_per_task['all'] = []
        _annos = []
        print(f'Loading {split} annotations...')
        for fname, seed, task in zip(filenames, seeds, task_names):
            _path = os.path.join(self._path, task + '-' + split)
            self.cache[task + '/' + fname] = {'seed': seed, 'task': task}
            self.seeds_per_task[task].append((task, seed))
            self.seeds_per_task['all'].append((task, seed))
            with open(os.path.join(_path, 'reward', fname), 'rb') as fid:
                rewards = pickle.load(fid)  # len(actions)
            if not (np.array(rewards)[1:] > 0).all():
                warnings.warn(f'WARNING: imperfect demo {fname} for {task}')
            pairs = []
            for k in range(len(rewards) - 1):
                if rewards[k + 1] == 0:  # unsuccessful action
                    continue
                pairs.append(k)
            _annos += [(fname, task, p) for p in range(len(pairs))]
            self.cache[task + '/' + fname]['obs_act'] = None
        self.annos = []
        for _ in range(repeat):  # repeat to support larger batch size
            self.annos += _annos

    @staticmethod
    def _action2point(action):
        p0_xyz, p0_xyzw = action['pose0']
        p1_xyz, p1_xyzw = action['pose1']
        p0 = utils.xyz_to_pix(p0_xyz, BOUNDS, PIXEL_SIZE)
        p0_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p0_xyzw)[2])
        p1 = utils.xyz_to_pix(p1_xyz, BOUNDS, PIXEL_SIZE)
        p1_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p1_xyzw)[2])
        p1_theta = p1_theta - p0_theta
        p0_theta = 0
        return p0, p1, p0_theta, p1_theta

    @staticmethod
    def _augment(ep, transform_params=None, theta_sigma=60):
        # ep is a dict from self.cache['obs_act']
        img, _, (p0, p1), boxes, transform_params = utils.perturb_wboxes(
            np.copy(ep['image']),
            [np.copy(ep['p0']), np.copy(ep['p1'])],
            np.concatenate((ep['pick_boxes'], ep['place_boxes'])),
            theta_sigma=theta_sigma,
            add_noise=True,
            transform_params=transform_params
        )
        return img, p0, p1, boxes, transform_params

    @staticmethod
    def _info2box(objects):
        boxes = []
        for value in objects.values():
            left = value[0][0]
            right = value[0][1]
            boxes.append([left[0], left[1], right[0], right[1]])
        return np.asarray(boxes).reshape(-1, 4)

    def add_to_cache(self, fname, task):
        _path = os.path.join(self._path, task + '-' + self.split)
        with open(os.path.join(_path, 'reward', fname), 'rb') as fid:
            rewards = pickle.load(fid)  # len(actions)
        with open(os.path.join(_path, 'color', fname), 'rb') as fid:
            color = pickle.load(fid)  # len(actions), 3, 480, 640, 3
        with open(os.path.join(_path, 'depth', fname), 'rb') as fid:
            depth = pickle.load(fid)  # len(actions), 3, 480, 640
        with open(os.path.join(_path, 'action', fname), 'rb') as fid:
            action = pickle.load(fid)  # len(actions) list of dicts
        with open(os.path.join(_path, 'info', fname), 'rb') as fid:
            info = pickle.load(fid)  # len(actions) list of dicts
        pairs = []
        for k in range(len(rewards) - 1):
            if rewards[k + 1] == 0:  # unsuccessful action
                continue
            p0, p1, p0_theta, p1_theta = self._action2point(action[k])
            if not self.cliport and 'move_goal' in info[k] and len(info[k]['move_goal']) != 0:
            # if 'move_goal' in info[k] and len(info[k]['move_goal']) != 0:
                place_boxes = self._info2box(info[k+1]['move_goal'])
            else:
                place_boxes = self._info2box(info[k]['fixed'])

            pairs.append({
                'lang_goal': info[k]['lang_goal'],
                'pick_boxes': self._info2box(info[k]['move']),
                'place_boxes': place_boxes,
                'goal_boxes': np.concatenate((
                    self._info2box(info[k + 1]['move']), # can change to 'move_goal' if we regenerate data for everything
                    self._info2box(info[k + 1]['fixed'])
                )),
                'color': color[k],
                'depth': depth[k],
                'action': action[k],
                'image': get_image(
                    {'color': color[k], 'depth': depth[k]},
                    square_pad=self.square_pad, reshape=self.reshape
                ),
                'goal_image': get_image(
                    {'color': color[k + 1], 'depth': depth[k + 1]},
                    square_pad=self.square_pad, reshape=self.reshape
                ),
                'p0': (
                    np.round(p0) if self.reshape is None
                    else np.round(np.copy(p0) * self.reshape / 640)
                ),
                'p1': (
                    np.round(p1) if self.reshape is None
                    else np.round(np.copy(p1) * self.reshape / 640)
                ),
                'p0_theta': p0_theta,
                'p1_theta': p1_theta
            })
        self.cache[task + '/' + fname]['obs_act'] = pairs

    @staticmethod
    def _boxes2map(boxes, shape):
        map_ = np.zeros(shape[:-1])
        for box in boxes:
            map_[int(box[0]):int(box[2]), int(box[1]):int(box[3])] = 1
        return map_

    def get_seed(self, idx):
        name, task, _ = self.annos[idx]
        return self.cache[task + '/' + name]['seed']

    def get_seed_by_task_and_idx(self, task, idx):
        return self.seeds_per_task[task][idx]

    def retrieve_by_task_and_name(self, task, name, obs_act_id=None, theta_sigma=False):
        if self.cache[task + '/' + name]['obs_act'] is None:
            self.add_to_cache(name, task)
        if obs_act_id is not None:
            anno = self.cache[task + '/' + name]['obs_act'][obs_act_id]
            if self.augment:
                img, p0, p1, boxes, _ = self._augment(anno, theta_sigma=theta_sigma)
                pick_boxes = boxes[:len(anno['pick_boxes'])]
                place_boxes = boxes[len(anno['pick_boxes']):]
            else:
                img, p0, p1 = anno['image'], anno['p0'], anno['p1']
                pick_boxes = anno['pick_boxes']
                place_boxes = anno['place_boxes']
        else:  # fetch all steps
            anno, img, p0, p1 = [], [], [], []
            transform_params = None
            for anno_ in self.cache[task + '/' + name]['obs_act']:
                anno.append(anno_)
                if self.augment:
                    img_, p0_, p1_, transform_params = self._augment(
                        anno_, transform_params, theta_sigma=theta_sigma
                    )
                else:
                    img_, p0_, p1_ = anno_['image'], anno_['p0'], anno_['p1']
                img.append(img_)
                p0.append(p0_)
                p1.append(p1_)
            img = np.stack(img)
            p0 = np.stack(p0)
            p1 = np.stack(p1)
        return anno, img, p0, p1, pick_boxes, place_boxes

    def _fetch_idx_by_task(self, task):
        _annos = [
            a
            for a, anno in enumerate(self.annos)
            if anno[1] == task
        ]
        idx = np.random.randint(0, len(_annos))
        return _annos[idx]

    def _custom_getitem(self, idx, prefix=''):
        name, task, obs_act_id = self.annos[idx]
        anno, img, p0, p1, pk_boxes, pl_boxes = self.retrieve_by_task_and_name(
            task, name, obs_act_id, theta_sigma=self.theta_sigma
        )  # <- this function leads to a seg fault!

        # do some padding to pick and place boxes to avoid errors due to rounding
        pk_boxes[:, :2] -= 4
        pk_boxes[:, 2:] += 4
        pl_boxes[:, :2] -= 4
        pl_boxes[:, 2:] += 4

        assert isinstance(anno, dict)
        assert len(anno['pick_boxes']) <= MAX_BOXES, f"More pick boxes: {len(anno['pick_boxes'])} than the limit"
        pick_bx = np.ones((MAX_BOXES, 4))
        pick_mask = np.zeros(MAX_BOXES)
        pick_bx[:len(anno['pick_boxes'])] = pk_boxes
        pick_mask[:len(anno['pick_boxes'])] = 1
        place_bx = np.ones((MAX_BOXES, 4))
        place_mask = np.zeros(MAX_BOXES)
        place_bx[:len(anno['place_boxes'])] = pl_boxes
        place_mask[:len(anno['place_boxes'])] = 1

        # find actual pick and place box for the current action
        pick_mask_ = ((pk_boxes[:, 0] <= p0[0]) & (pk_boxes[:, 2] >= p0[0]) & (pk_boxes[:, 1] <= p0[1]) & (pk_boxes[:, 3] >= p0[1])).astype(np.bool8)

        if pick_mask_.sum() == 0:
            assert False

        place_mask_ = ((pl_boxes[:, 0] <= p1[0]) & (pl_boxes[:, 2] >= p1[0]) & (pl_boxes[:, 1] <= p1[1]) & (pl_boxes[:, 3] >= p1[1])).astype(np.bool8)
        if place_mask_.sum() == 0:
            assert False

        gt_pick_box = pk_boxes[pick_mask_.argmax()]
        gt_place_box = pl_boxes[place_mask_.argmax()]

        return {
            prefix + 'image': img,  # (640, 320, 6)
            prefix + 'p0': np.copy(p0),  # (y, x)
            prefix + 'p1': np.copy(p1),  # (y, x)
            prefix + 'p0_theta': anno['p0_theta'],
            prefix + 'p1_theta': anno['p1_theta'],
            prefix + 'lang_goal': anno['lang_goal'],
            prefix + 'pick_nodes': pick_bx,  # (y1, x1, y2, x2)
            prefix + 'place_nodes': place_bx,  # (y1, x1, y2, x2)
            prefix + 'pick_mask': np.array(pick_mask),
            prefix + 'place_mask': np.array(place_mask),
            prefix + 'task': task,
            prefix + 'gt_pick_box': gt_pick_box,
            prefix + 'gt_place_box': gt_place_box,
        }

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        if self.analogical:
            ret_dict = self._custom_getitem(idx, 't_')
            anchor_idx = self._fetch_idx_by_task(ret_dict['t_task'])
            if np.random.random() < 0.15 and self.split == 'train':
                anchor_idx = idx
            ret_dict.update(self._custom_getitem(anchor_idx, 'mem_'))
        else:
            ret_dict = self._custom_getitem(idx, '')
        return ret_dict


def get_image(obs, cam_config=None, square_pad=False, reshape=None, bounds=None, pixel_size=None):
    """
    Stack color and height images image.

    obs = {  # from 3 cameras
        'color': (front (480, 640, 3), left, right),
        'depth': (front (480, 640), left, right)
    }
    """
    if cam_config is None:
        cam_config = CAMERA_CONFIG

    if bounds is None:
        bounds = BOUNDS

    if pixel_size is None:
        pixel_size = PIXEL_SIZE

    # Get color and height maps from RGB-D images.
    cmap, hmap = utils.get_fused_heightmap(
        obs, cam_config, bounds, pixel_size
    )
    img = np.concatenate((
        cmap,
        hmap[..., None],
        hmap[..., None],
        hmap[..., None]
    ), axis=2)
    if square_pad:
        diff = img.shape[1] - img.shape[0]
        if diff > 0:
            img = np.concatenate((
                img,
                np.zeros((diff, img.shape[1], img.shape[2]))
            ))
        elif diff < 0:
            img = np.concatenate((
                img,
                np.zeros((img.shape[0], -diff, img.shape[2]))
            ), axis=1)
    if reshape is not None:
        img = np.concatenate((
            cv2.resize(img[..., :3], (reshape, reshape)),
            cv2.resize(img[..., 3:], (reshape, reshape))
        ), axis=2)
    return img
