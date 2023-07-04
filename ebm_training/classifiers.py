import torch


def rightcls(inputs):
    # (N, 2)
    if len(inputs) == 5:
        sbox_centers, sbox_sizes, obox_centers, obox_sizes, _ = inputs
    else:
        sbox_centers, sbox_sizes, obox_centers, obox_sizes = inputs
    cond = sbox_centers - sbox_sizes * 0.5 > obox_centers + obox_sizes * 0.5
    return cond[:, 0].float()


def leftcls(inputs):
    # (N, 2)
    if len(inputs) == 5:
        sbox_centers, sbox_sizes, obox_centers, obox_sizes, _ = inputs
    else:
        sbox_centers, sbox_sizes, obox_centers, obox_sizes = inputs
    cond = sbox_centers + sbox_sizes * 0.5 < obox_centers - obox_sizes * 0.5
    return cond[:, 0].float()


def frontcls(inputs):
    # (N, 2)
    if len(inputs) == 5:
        sbox_centers, sbox_sizes, obox_centers, obox_sizes, _ = inputs
    else:
        sbox_centers, sbox_sizes, obox_centers, obox_sizes = inputs
    cond = sbox_centers + sbox_sizes * 0.5 < obox_centers - obox_sizes * 0.5
    return cond[:, 1].float()


def behindcls(inputs):
    # (N, 2)
    if len(inputs) == 5:
        sbox_centers, sbox_sizes, obox_centers, obox_sizes, _ = inputs
    else:
        sbox_centers, sbox_sizes, obox_centers, obox_sizes = inputs
    cond = sbox_centers - sbox_sizes * 0.5 > obox_centers + obox_sizes * 0.5
    return cond[:, 1].float()


def inscls(inputs):
    # (N, 2)
    if len(inputs) == 5:
        sbox_centers, sbox_sizes, obox_centers, obox_sizes, _ = inputs
    else:
        sbox_centers, sbox_sizes, obox_centers, obox_sizes = inputs
    x1 = sbox_centers[:, 0]
    y1 = sbox_centers[:, 1]
    w1 = sbox_sizes[:, 0]
    h1 = sbox_sizes[:, 1]
    x2 = obox_centers[:, 0]
    y2 = obox_centers[:, 1]
    w2 = obox_sizes[:, 0]
    h2 = obox_sizes[:, 1]
    cond = (
        (x1 - w1 * 0.5 > x2 - w2 * 0.5)
        & (x1 + w1 * 0.5 < x2 + w2 * 0.5)
        & (y1 - h1 * 0.5 > y2 - h2 * 0.5)
        & (y1 + h1 * 0.5 < y2 + h2 * 0.5)
    )
    intersection = (
        (
            torch.minimum(x1 + w1 * 0.5, x2 + w2 * 0.5)
            - torch.maximum(x1 - w1 * 0.5, x2 - w2 * 0.5)
        )
        * (
            torch.minimum(y1 + h1 * 0.5, y2 + h2 * 0.5)
            - torch.maximum(y1 - h1 * 0.5, y2 - h2 * 0.5)
        )
    )
    area = w1 * h1
    cond = intersection / area > 0.8
    return cond.float()


def ccls(inputs):
    # (B, N, 2)
    if len(inputs) == 5:
        sbox_centers, sbox_sizes, obox_centers, obox_sizes, _ = inputs
    else:
        sbox_centers, sbox_sizes, obox_centers, mask = inputs
    centers = sbox_centers  # .detach().cpu().numpy()
    # Find centroid
    centroid = (centers * mask[..., None]).sum(1) / mask.sum(1)[:, None]
    # Check if all points fall between [0.9r, 1.1r]
    dists = torch.sqrt(torch.sum(
        ((centers - centroid[:, None]) ** 2) * mask[..., None], -1
    ))  # B N
    # Find radius
    radius = (dists * mask).sum(1) / mask.sum(1)
    ret = (
        (dists > 0.95 * radius[:, None]) & (dists < 1.05 * radius[:, None])
    ).float().sum(1)
    return (ret == mask.sum(1)).float()


CLASSIFIERS = {
    'right': rightcls,
    'left': leftcls,
    'front': frontcls,
    'behind': behindcls,
    'inside': inscls
}
