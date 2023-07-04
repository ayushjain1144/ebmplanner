import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2

class ReservoirBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of samples to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.n = 0

    def __len__(self):
        return len(self._storage)

    def add(self, ims):
        batch_size = ims.shape[0]
        if self._next_idx >= len(self._storage):
            self._storage.extend(list(ims))
            self.n = self.n + ims.shape[0]
        else:
            for im in ims:
                self.n = self.n + 1
                ix = random.randint(0, self.n - 1)

                if ix < len(self._storage):
                    self._storage[ix] = im

        self._next_idx = (self._next_idx + ims.shape[0]) % self._maxsize


    def _encode_sample(self, idxes, no_transform=False, downsample=False):
        ims = []
        for i in idxes:
            im = self._storage[i]

            # if self.dataset != "mnist":
            #     if (self.transform is not None) and (not no_transform):
            #         im = im.transpose((1, 2, 0))
            #         im = np.array(self.transform(Image.fromarray(im)))

            #     # if downsample and (self.dataset in ["celeba", "object", "imagenet"]):
            #     #     im = im[:, ::4, ::4]

            # im = im * 255

            ims.append(im)
        return np.array(ims)

    def sample(self, batch_size, no_transform=False, downsample=False):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1)
                 for _ in range(batch_size)]
        return self._encode_sample(idxes, no_transform=no_transform, downsample=downsample), idxes
