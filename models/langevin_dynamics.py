"""Langevin dynamics for concept learning with EBMs."""

import torch


XMIN = -1
XMAX = 1
YMIN = -0.5
YMAX = 0.5
ZMIN = -1
ZMAX = 1
SIZE = 0.15

LENS = torch.tensor(
    [XMAX - XMIN - SIZE, YMAX - YMIN - SIZE, ZMAX - ZMIN - SIZE]
) / 2.0

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def run_model(models, centers, sizes, subj, obj, rel, move_all):
    energy = 0
    for k in range(len(subj)):
        s = subj[k]
        o = obj[k]
        r = rel[k]
        if r in ['circle', 'square', 'triangle', 'line']:
            energy = energy + models[r]((
                centers[torch.as_tensor(s)][None],
                None,
                None,
                torch.ones(1, len(s)).to(DEVICE)
            ))
            continue
        for s_ in s:
            for o_ in o:
                if move_all:
                    energy = energy + models[r]((
                        centers[s_].view(1, 2),
                        sizes[s_].view(1, 2),
                        centers[o_].view(1, 2),
                        sizes[o_].view(1, 2)
                    )) / (len(s) * len(o))
                else:
                    energy = energy + models[r]((
                        centers[s_].view(1, 2),
                        sizes[s_].view(1, 2),
                        centers[o_].view(1, 2).detach(),
                        sizes[o_].view(1, 2).detach()
                    )) / (len(s) * len(o))
    return energy


def langevin(models, boxes, subj, obj, rel, move_all=True):
    boxes = torch.from_numpy(boxes).float()
    centers = boxes[..., :2].to(DEVICE)
    sizes = boxes[..., 2:].to(DEVICE)
    noise = torch.randn_like(centers).detach()
    negs_samples = []

    for _ in range(50):
        # Add noise
        noise.normal_()
        # centers = centers + 0.005 * noise

        # Forward pass
        centers.requires_grad_(requires_grad=True)
        energy = run_model(models, centers, sizes, subj, obj, rel, move_all)

        # Backward pass (gradients wrt image)
        _grad = torch.autograd.grad([energy.sum()], [centers])[0]
        centers = centers - _grad

        # Detach/clamp/store
        centers = centers.detach()
        centers[..., 0] = torch.clamp(
            centers[..., 0],
            XMIN + sizes[..., 0] / 2, XMAX - sizes[..., 0] / 2
        )
        centers[..., 1] = torch.clamp(
            centers[..., 1],
            YMIN + sizes[..., 1] / 2, YMAX - sizes[..., 1] / 2
        )
        negs_samples.append(torch.cat((centers, sizes), -1))

    return torch.cat((centers, sizes), -1), negs_samples
