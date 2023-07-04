import cv2
import numpy as np
import torch
from torch import functional as F


def back2color(i, blacken_zeros=False):
    if blacken_zeros:
        const = torch.tensor([-0.5])
        i = torch.where(i == 0.0, const.cuda() if i.is_cuda else const, i)
        return back2color(i)
    else:
        return ((i + 0.5) * 255).type(torch.ByteTensor)


def draw_frame_id_on_vis(vis, frame_id):

    rgb = vis.detach().cpu().numpy()[0]
    rgb = np.transpose(rgb, [1, 2, 0])  # put channels last
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    color = (255, 255, 255)

    def strnum(x):
        s = '%g' % x
        if '.' in s:
            s = s[s.index('.'):]
        return s
    frame_str = strnum(frame_id)

    cv2.putText(
        rgb,
        frame_str,
        (5, 20),  # from left, from top
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,  # font scale (float)
        color,
        1
    )  # font thickness (int)
    rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
    vis = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return vis


class Summ_writer:

    def __init__(self, writer, global_step, log_freq=10, fps=8):
        self.writer = writer
        self.global_step = global_step
        self.log_freq = log_freq
        self.fps = fps
        self.maxwidth = 1800
        self.save_this = (self.global_step % self.log_freq == 0)

    def summ_gif(self, name, tensor, blacken_zeros=False):
        # tensor should be in B x S x C x H x W

        if tensor.dtype == torch.float32:
            tensor = back2color(tensor, blacken_zeros=blacken_zeros)

        video_to_write = tensor[:1]

        self.writer.add_video(
            name, video_to_write,
            fps=self.fps, global_step=self.global_step
        )
        return video_to_write

    def summ_rgb(self, name, ims, blacken_zeros=False, frame_id=None,
                 only_return=False, halfres=False):
        if self.save_this:
            assert ims.dtype in {torch.uint8, torch.float32}

            if ims.dtype == torch.float32:
                ims = back2color(ims, blacken_zeros)

            # ims is B x C x H x W
            vis = ims[:1]  # just the first one
            B, C, H, W = list(vis.shape)

            if halfres:
                vis = F.interpolate(vis, scale_factor=0.5)

            if frame_id is not None:
                vis = draw_frame_id_on_vis(vis, frame_id)

            if int(W) > self.maxwidth:
                vis = vis[:, :, :, :self.maxwidth]

            if only_return:
                return vis
            else:
                return self.summ_gif(name, vis.unsqueeze(1), blacken_zeros)
