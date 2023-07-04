import numpy as np
import torch
from torch import nn

from global_vars import BOUNDS, PIXEL_SIZE, IN_SHAPE
from models.transporter_blocks import Attention, Transport
import utils.transporter_utils as utils


class TransporterAgent(nn.Module):
    """Base class to implement Transporter in PyTorch."""

    def __init__(self, n_rotations, cfg):
        """
        Args:
            - n_rotations (int): number of orientation bins for
                end-effectors pose
        """
        super().__init__()
        utils.set_seed(0)

        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.crop_size = 64
        self.n_rotations = n_rotations  # 36
        self.cfg = cfg

        self.pix_size = PIXEL_SIZE
        self.in_shape = IN_SHAPE
        self.bounds = BOUNDS
        self.real_robot = False

        self._build_model()

    def _build_model(self):
        self.pick_net = None
        self.place_net = None

    def pick_forward(self, img, lang=None, softmax=False):
        return self.pick_net(img, lang, softmax=softmax)

    def place_forward(self, img, pick_pose, lang=None, softmax=False, img_pick=None):
        return self.place_net(img, pick_pose, lang, softmax=softmax, img_pick=img_pick)

    def forward(self, img, lang=None, pick_hmap=None, place_hmap=None):
        """Run an inference step given visual observations (H, W, 6)."""
        # Pick model forward pass
        pick_conf = self.pick_forward(img.unsqueeze(0), lang, True)[0]
        pick_conf = pick_conf.detach().cpu()
        tr_pick = np.copy(pick_conf.numpy())
        if pick_hmap is not None:
            pick_conf = pick_conf * pick_hmap[..., None]
            pick_conf = pick_conf + 0.1 * pick_hmap[..., None]
        argmax = np.argmax(pick_conf.numpy())
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0_pix = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

        # Place model forward pass
        place_conf = self.place_forward(
            img.unsqueeze(0),
            (torch.as_tensor([p0_pix[0]]), torch.as_tensor([p0_pix[1]])),
            lang,
            softmax=True
        )[0]
        place_conf = place_conf.detach().cpu()
        tr_place = np.copy(place_conf.numpy())
        if place_hmap is not None:
            place_conf = place_conf * place_hmap[..., None]
            place_conf = place_conf + 0.1 * place_hmap[..., None]

        argmax = np.argmax(place_conf.numpy())
        argmax = np.unravel_index(argmax, shape=place_conf.shape)
        p1_pix = argmax[:2]
        p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

        if self.real_robot:
            p0_pix = torch.tensor([p0_pix[1], p0_pix[0]])
            p1_pix = torch.tensor([p1_pix[1], p1_pix[0]])
            height = height.transpose(1, 0)

        # Pixels to end effector poses
        height = img[..., 3].cpu().numpy()
        p0_xyz = utils.pix_to_xyz(p0_pix, height, self.bounds, self.pix_size)
        p1_xyz = utils.pix_to_xyz(p1_pix, height, self.bounds, self.pix_size)
        p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
        p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))
        return {
            'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
            'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
            'pick': p0_pix,
            'place': p1_pix
        }, pick_conf, place_conf, tr_pick, tr_place


class OriginalTransporterAgent(TransporterAgent):

    def __init__(self, n_rotations, cfg, pretrained=False):
        super().__init__(n_rotations, cfg)
        stream_fcn = 'plain_resnet'
        self.pick_net = Attention(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
            pretrained=pretrained
        )
        self.place_net = Transport(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
            pretrained=pretrained
        )


class GCTransporterAgent(TransporterAgent):

    def __init__(self, n_rotations, cfg, pretrained=False):
        super().__init__(n_rotations, cfg)
        stream_fcn = 'plain_resnet'
        self.pick_net = Attention(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
            pretrained=pretrained
        )
        self.place_net = Transport(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
            pretrained=pretrained
        )

    def forward(self, img_pick, img_place):
        """Run an inference step given visual observations (H, W, 6)."""
        # Pick model forward pass
        pick_conf = self.pick_forward(img_pick.unsqueeze(0), None, True)[0]
        pick_conf = pick_conf.detach().cpu()
        tr_pick = np.copy(pick_conf.numpy())

        argmax = np.argmax(pick_conf.numpy())
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0_pix = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

        # Place model forward pass
        place_conf = self.place_forward(
            img_place.unsqueeze(0),
            (torch.as_tensor([p0_pix[0]]), torch.as_tensor([p0_pix[1]])),
            lang=None,
            softmax=True,
            img_pick=img_pick.unsqueeze(0)
        )[0]
        place_conf = place_conf.detach().cpu()
        tr_place = np.copy(place_conf.numpy())

        argmax = np.argmax(place_conf.numpy())
        argmax = np.unravel_index(argmax, shape=place_conf.shape)
        p1_pix = argmax[:2]
        p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

        # Pixels to end effector poses
        height_pick = img_pick[..., 3].cpu().numpy()
        height_place = img_place[..., 3].cpu().numpy()
        if self.real_robot:
            p0_pix = torch.tensor([p0_pix[1], p0_pix[0]])
            p1_pix = torch.tensor([p1_pix[1], p1_pix[0]])

            p0_pix = torch.where(img_pick[..., :2])
            p0_pix = torch.tensor(
                [p0_pix[1].float().mean().to(torch.int64),
                 p0_pix[0].float().mean().to(torch.int64)])

            p1_pix = torch.where(img_place[..., :2])
            p1_pix = torch.tensor([
                p1_pix[1].float().mean().to(torch.int64),
                p1_pix[0].float().mean().to(torch.int64)
            ])

            height_pick = height_pick.transpose(1, 0)
            height_place = height_place.transpose(1, 0)

        p0_xyz = utils.pix_to_xyz(p0_pix, height_pick, self.bounds, self.pix_size)
        p1_xyz = utils.pix_to_xyz(p1_pix, height_place, self.bounds, self.pix_size)
        p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
        p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))
        return {
            'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
            'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
            'pick': p0_pix,
            'place': p1_pix,
        }, pick_conf, place_conf, tr_pick, tr_place
