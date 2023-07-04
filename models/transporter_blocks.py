"""Attention module."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

import models
import utils.transporter_utils as utils


class Attention(nn.Module):
    """Attention (a.k.a Pick) module."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg,
                 device, pretrained=False):
        """
        Args:
            - stream_fcn (tuple): (str, str or None), names of fcn to use
            - in_shape (tuple): e.g. (640, 320, 6)
            - n_rotations (int): number of orientation bins
            - preprocess (func): a preprocessing function
            - cfg (dict): configuration file
            - device (str): torch device to use (cuda or cpu)
        """
        super().__init__()
        self.padding = np.zeros((3, 2), dtype=int)
        max_dim = np.max(in_shape[:2])
        pad = (max_dim - np.array(in_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)

        in_shape = np.array(in_shape)
        in_shape += np.sum(self.padding, axis=1)
        in_shape = tuple(in_shape)

        self.stream_fcn = stream_fcn
        self.in_shape = in_shape
        self.preprocess = preprocess
        self.cfg = cfg
        self.device = device
        self.pretrained = pretrained

        self._build_nets()

    def _build_nets(self):
        if not self.pretrained:
            stream_one_fcn, _ = self.stream_fcn
            print(f"Attention FCN: {stream_one_fcn}")
            self.attn_stream = models.names[stream_one_fcn](
                self.in_shape, 1, self.cfg, self.device, self.preprocess
            )
            print(f"Attn FCN: {stream_one_fcn}")
        else:
            self.attn_stream = PretrainedResNet()
            print('pretrained attention')

    def attend(self, x, lang=None):
        """Input should be (B, C, H, W)."""
        return self.attn_stream(x)

    def _pad(self, img):
        padding = self.padding[:2][::-1].T.flatten().tolist()  # (l, t, r, b)
        img = img.permute(0, 3, 1, 2)
        return transforms.Pad(padding)(img)

    def forward(self, inp_img, lang=None, goal_img=None, softmax=False):
        """Forward pass, inp_img is (B, H, W, 6), output (B, H, W, nrot)."""
        in_tens = self._pad(inp_img)

        if goal_img is not None:
            goal_tensor = self._pad(goal_img)
            in_tens = in_tens * goal_tensor

        # Forward pass
        logits = self.attend(in_tens, lang).unsqueeze(1)

        # Un-pad back output
        c0 = self.padding[:2, 0]
        c1 = c0 + inp_img.shape[1:3]
        logits = logits[:, :, :, c0[0]:c1[0], c0[1]:c1[1]]

        # Reshape and normalize
        logits = logits.squeeze(2).permute(0, 2, 3, 1)  # [B H W nrot]
        if softmax:
            output = logits.reshape(len(logits), -1)
            output = F.softmax(output, dim=-1)
            output = output.reshape(logits.shape)
        else:
            output = logits
        return output


class PretrainedResNet(nn.Module):
    """Wrapper for pre-trained ResNet-FCN for RGBD."""

    def __init__(self, num_classes=1):
        super().__init__()
        self.net = deeplabv3_mobilenet_v3_large(pretrained=True).eval()
        # Replace first layer
        new_layer = nn.Conv2d(4, 16, 3, stride=2, padding=(1, 1), bias=False)
        new_layer.weight.data[:, :3] = self.net.backbone['0'][0].weight
        self.net.backbone['0'][0] = new_layer
        # Replace output layer
        self.net.classifier[4] = torch.nn.Conv2d(256, num_classes, 1)
        # Normalization factors
        self.color = torch.as_tensor([255, 255, 255, 1.0])[:, None, None]
        d_m, d_s = 0.00509261, 0.00903967  # depth meand/std
        self.mean = torch.as_tensor([0.485, 0.456, 0.406, d_m])[:, None, None]
        self.std = torch.as_tensor([0.229, 0.224, 0.225, d_s])[:, None, None]

    def forward(self, img):
        """Forward pass, img is (B, 4, H, W)."""
        img = img[:, :4] / self.color.to(img.device)[None]
        img = img - self.mean.to(img.device)[None]
        img = img / self.std.to(img.device)[None]
        return self.net(img)['out']

    def train(self, mode=True):
        """Sets always in eval mode."""
        mode = False
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self


class Transport(nn.Module):
    """Transport (a.k.a Place) module."""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size,
                 preprocess, cfg, device, goal_conditioned=False,
                 pretrained=False):
        """
        Args:
            - stream_fcn (tuple): (str, str or None), names of fcn to use
            - in_shape (tuple): e.g. (640, 320, 6)
            - n_rotations (int): number of orientation bins
            - crop_size (int): side of square crop around pick, must be N*16
            - preprocess (func): a preprocessing function
            - cfg (dict): configuration file
            - device (str): torch device to use (cuda or cpu)
        """
        super().__init__()
        self.n_rotations = n_rotations
        self.goal_conditioned = goal_conditioned

        self.pad_size = int(crop_size / 2)
        self.padding = np.zeros((3, 2), dtype=int)
        self.padding[:2, :] = self.pad_size
        in_shape = tuple(in_shape)

        # Crop before network (default from Transporters CoRL 2020)
        self.kernel_shape = (crop_size, crop_size, in_shape[2])
        self.output_dim = 3
        self.kernel_dim = 3
        self.rotator = utils.ImageRotatorBatched(self.n_rotations)

        self.stream_fcn = stream_fcn
        self.in_shape = in_shape
        self.preprocess = preprocess
        self.cfg = cfg
        self.device = device
        self.pretrained = pretrained

        self._build_nets()

    def _build_nets(self):
        if not self.pretrained:
            stream_one_fcn, _ = self.stream_fcn
            model = models.names[stream_one_fcn]
            self.key_resnet = model(self.in_shape, self.output_dim,
                                    self.cfg, self.device, self.preprocess)
            self.query_resnet = model(self.kernel_shape, self.kernel_dim,
                                      self.cfg, self.device, self.preprocess)
            if self.goal_conditioned:
                self.g_net = model(self.in_shape, self.output_dim,
                                   self.cfg, self.device, self.preprocess)
            print(f"Transport FCN: {stream_one_fcn}")
        else:
            self.key_resnet = PretrainedResNet(self.output_dim)
            self.query_resnet = PretrainedResNet(self.kernel_dim)
            print('pretrained transport')

    def correlate(self, in0, in1, softmax):
        """Correlate two input tensors."""
        output = F.conv2d(in0, in1, padding=(self.pad_size, self.pad_size))
        output = F.interpolate(
            output, size=(in0.shape[-2], in0.shape[-1]),
            mode='bilinear', align_corners=False
        )
        output = output[
            :, :, self.pad_size:-self.pad_size, self.pad_size:-self.pad_size
        ]
        if softmax:
            output_shape = output.shape
            output = output.reshape(len(output), -1)
            output = F.softmax(output, dim=-1)
            output = output.reshape(output_shape)
        return output[0]

    def transport(self, in_tensor, crop, lang=None):
        logits = self.key_resnet(in_tensor)
        kernel = self.query_resnet(crop)
        return logits, kernel

    def goal_transport(self, in_tensor, crop):
        logits = self.g_net(in_tensor)
        kernel = self.g_net(crop)
        return logits, kernel

    def _pad(self, img):
        # img is (B, H, W, C)
        padding = self.padding[:2][::-1].T.flatten().tolist()  # (l, t, r, b)
        img = img.permute(0, 3, 1, 2)
        return transforms.Pad(padding)(img)

    def _crop(self, img, pv):
        # img is (B, C, H, W)
        hcrop = self.pad_size
        crop = img.unsqueeze(1).repeat(1, self.n_rotations, 1, 1, 1)
        crops = []
        for b in range(len(crop)):
            crop_ = self.rotator(crop[b].unsqueeze(0), pivot=pv[b])
            crops.append(crop_[
                0, :, :,
                pv[b, 0]-hcrop:pv[b, 0]+hcrop,
                pv[b, 1]-hcrop:pv[b, 1]+hcrop
            ])
        return torch.stack(crops)

    def forward(
        self, inp_img, p, lang=None,
        goal_img=None, softmax=False, img_pick=None
    ):
        """
        Forward pass.

        Args:
            - inp_img (tensor): (B, H, W, 6)
            - p (tuple of tensor): pick position (y, x), ((B,), (B,))

        Returns:
            - place map (tensor): (B, H, W, n_rotations)
        """
        in_tens = self._pad(inp_img)
        if self.goal_conditioned and goal_img is not None:
            goal_tens = self._pad(goal_img)

        # Rotation pivot
        pv = torch.stack([p[0], p[1]], 1) + self.pad_size

        # Crop before network (default from Transporters CoRL 2020)
        if img_pick is not None:
            pick_tens = self._pad(img_pick)
            crop = self._crop(pick_tens, pv)
        else:
            crop = self._crop(in_tens, pv)
        if self.goal_conditioned and goal_img is not None:
            goal_crop = self._crop(goal_tens, pv)

        logits, kernel = self.transport(
            in_tens,
            crop.reshape((-1,) + crop.shape[2:]),
            lang
        )
        kernel = kernel.reshape(crop.shape[:2] + (3,) + crop.shape[3:])
        if self.goal_conditioned and goal_img is not None:
            g_logits, g_kernel = self.goal_transport(
                goal_tens,
                goal_crop.reshape((-1,) + goal_crop.shape[2:])
            )
            g_kernel = g_kernel.reshape(
                goal_crop.shape[:2] + (3,) + goal_crop.shape[3:]
            )
            logits = logits + g_logits
            kernel = kernel + g_kernel

        return torch.stack([
            self.correlate(lgt.unsqueeze(0), kern, softmax)
            for lgt, kern in zip(logits, kernel)
        ]).permute(0, 2, 3, 1).contiguous()
