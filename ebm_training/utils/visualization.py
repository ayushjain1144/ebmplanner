from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
import io
from PIL import Image
import torch


def _corner_points(box):
    """Convert min-max box to corner points."""
    [xmin, ymin, zmin, xmax, ymax, zmax] = box
    c1 = np.array([xmin, ymin, zmin])
    c2 = np.array([xmax, ymin, zmin])
    c3 = np.array([xmax, ymax, zmin])
    c4 = np.array([xmin, ymax, zmin])
    c5 = np.array([xmin, ymin, zmax])
    c6 = np.array([xmax, ymin, zmax])
    c7 = np.array([xmax, ymax, zmax])
    c8 = np.array([xmin, ymax, zmax])
    return np.vstack([c1, c2, c3, c4, c5, c6, c7, c8])


def plot_box(box, axis, c):
    """Plot a box with a given color."""
    box = np.concatenate((box[:3] - box[3:] / 2, box[:3] + box[3:] / 2))
    # print(box)
    corners = _corner_points(box)
    axis.plot(
        [corners[0, 0], corners[1, 0]],
        [corners[0, 1], corners[1, 1]],
        zs=[corners[0, 2], corners[1, 2]],
        c=c
    )
    axis.plot(
        [corners[1, 0], corners[2, 0]],
        [corners[1, 1], corners[2, 1]],
        zs=[corners[1, 2], corners[2, 2]],
        c=c
    )
    axis.plot(
        [corners[2, 0], corners[3, 0]],
        [corners[2, 1], corners[3, 1]],
        zs=[corners[2, 2], corners[3, 2]],
        c=c
    )
    axis.plot(
        [corners[3, 0], corners[0, 0]],
        [corners[3, 1], corners[0, 1]],
        zs=[corners[3, 2], corners[0, 2]],
        c=c
    )
    axis.plot(
        [corners[4, 0], corners[5, 0]],
        [corners[4, 1], corners[5, 1]],
        zs=[corners[4, 2], corners[5, 2]],
        c=c
    )
    axis.plot(
        [corners[5, 0], corners[6, 0]],
        [corners[5, 1], corners[6, 1]],
        zs=[corners[5, 2], corners[6, 2]],
        c=c
    )
    axis.plot(
        [corners[6, 0], corners[7, 0]],
        [corners[6, 1], corners[7, 1]],
        zs=[corners[6, 2], corners[7, 2]],
        c=c
    )
    axis.plot(
        [corners[7, 0], corners[4, 0]],
        [corners[7, 1], corners[0, 1]],
        zs=[corners[7, 2], corners[4, 2]],
        c=c
    )
    axis.plot(
        [corners[0, 0], corners[4, 0]],
        [corners[0, 1], corners[4, 1]],
        zs=[corners[0, 2], corners[4, 2]],
        c=c
    )
    axis.plot(
        [corners[1, 0], corners[5, 0]],
        [corners[1, 1], corners[5, 1]],
        zs=[corners[1, 2], corners[5, 2]],
        c=c
    )
    axis.plot(
        [corners[2, 0], corners[6, 0]],
        [corners[2, 1], corners[6, 1]],
        zs=[corners[2, 2], corners[6, 2]],
        c=c
    )
    axis.plot(
        [corners[3, 0], corners[7, 0]],
        [corners[3, 1], corners[7, 1]],
        zs=[corners[3, 2], corners[7, 2]],
        c=c
    )

    return axis


def plot_relations_3d(rel_boxes, ref_boxes, return_plt=False):
    """Plot a relation, rel/ref_boxes are (6,) center-size."""
    # Initialize figure
    fig = plt.figure(figsize=(12, 12))
    axis = fig.add_subplot(1, 1, 1, projection='3d')
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)
    axis.set_zlim(-1, 1)

    axis = plot_box(rel_boxes, axis, 'r')
    axis = plot_box(ref_boxes, axis, 'g')
    if return_plt:
        return plt

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = np.array(Image.open(buf))  # H x W x 4
    image = image[:, :, :3]
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).unsqueeze(0)  # 1 x 3 x H x W
    plt.close('all')
    return image


def plot_relations_2d(rel_boxes, ref_boxes, return_plt=False):
    """Plot a relation, rel/ref_boxes are (4,) or (6,) center-size."""
    if len(rel_boxes) == 6:
        rel_boxes = rel_boxes[(0, 1, 3, 4), ]
    if len(ref_boxes) == 6:
        ref_boxes = ref_boxes[(0, 1, 3, 4), ]
    # Initialize figure
    fig = plt.figure(figsize=(12, 12))
    axis = fig.add_subplot(1, 1, 1)
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)

    box = rel_boxes
    rect = patches.Rectangle(
        (box[0] - box[2] * 0.5, box[1] - box[3] * 0.5), box[2], box[3],
        linewidth=2, edgecolor='r', facecolor='none'
    )
    axis.add_patch(rect)
    box = ref_boxes
    rect = patches.Rectangle(
        (box[0] - box[2] * 0.5, box[1] - box[3] * 0.5), box[2], box[3],
        linewidth=2, edgecolor='g', facecolor='none'
    )
    axis.add_patch(rect)
    if return_plt:
        return plt

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = np.array(Image.open(buf))  # H x W x 4
    image = image[:, :, :3]
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).unsqueeze(0)  # 1 x 3 x H x W
    plt.close('all')
    return image


def plot_boxes_3d_all(ref_attention, ref_boxes, rel_attention, rel_boxes):
    """Plot ref and rel boxes together."""
    # boxes: [B, N, 6], attention: [B, N]
    ref_attention = ref_attention[0]
    ref_boxes = ref_boxes[0]
    rel_attention = rel_attention[0]
    rel_boxes = rel_boxes[0]

    # Initialize figure
    fig = plt.figure(figsize=(12, 12))
    axis = fig.add_subplot(1, 1, 1, projection='3d')
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)
    axis.set_zlim(-1, 1)

    # Plot ref boxes
    for i, box in enumerate(ref_boxes):
        if box.sum() == 0:
            continue
        if ref_attention[i] > 0.5:
            axis = plot_box(box, axis, 'b')
        else:
            axis = plot_box(box, axis, 'y')

    # Plot rel boxes
    for i, box in enumerate(rel_boxes):
        if box.sum() == 0:
            continue
        if rel_attention[i] > 0.5:
            axis = plot_box(box, axis, 'g')
        else:
            axis = plot_box(box, axis, 'r')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image = np.array(Image.open(buf))  # H x W x 4
    image = image[:, :, :3]
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).unsqueeze(0)  # 1 x 3 x H x W
    return image


def plot_boxes_2d_all(ref_attention, ref_boxes, return_plt=False):
    """Plot ref and rel boxes together."""
    # boxes: [N, 6], attention: [N]
    if len(ref_boxes) == 6:
        ref_boxes = ref_boxes[:, (0, 1, 3, 4)]
    # Initialize figure
    fig = plt.figure(figsize=(12, 12))
    axis = fig.add_subplot(1, 1, 1)
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)

    # Plot ref boxes
    for i, box in enumerate(ref_boxes):
        if box.sum() == 0:
            continue
        if ref_attention[i] > 0.5:
            rect = patches.Rectangle(
                (box[0] - box[2] * 0.5, box[1] - box[3] * 0.5), box[2], box[3],
                linewidth=2, edgecolor='g', facecolor='none'
            )
            axis.add_patch(rect)
    if return_plt:
        return plt

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image = np.array(Image.open(buf))  # H x W x 4
    image = image[:, :, :3]
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).unsqueeze(0)  # 1 x 3 x H x W
    return image


def plot_energy_2d(X, Y, z, centers):
    plt.figure(figsize=(12, 12))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.pcolormesh(X.numpy(), Y.numpy(), z.cpu().numpy(), shading='auto')
    for cent in centers[0, 1:].cpu().numpy():
        plt.scatter(cent[0], cent[1], s=100, c='r')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = np.array(Image.open(buf))  # H x W x 4
    image = image[:, :, :3]
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).unsqueeze(0)  # 1 x 3 x H x W
    return image


def plot_pose(centers, angles, targets, name='pose.jpg',
              show_line=True, show_center=True, show_target=True,
              return_buf=False):
    # centers (N, 2)
    x, y = centers[:, 0], centers[:, 1]
    # angles (N,)
    # targets (N, 2)
    r = np.sqrt(((targets - centers[:, :2]) ** 2).sum(-1))

    # find end point
    end_point = np.stack([x + r * np.cos(angles), y + r * np.sin(angles)], 1)
    end_point = np.clip(end_point, -0.85, 0.85)
    dir2center = np.stack((np.cos(angles), np.sin(angles)), 1)
    dir2tan = np.stack((
        np.cos(angles + np.pi / 2), np.sin(angles + np.pi / 2)
    ), 1)
    point1 = centers[:, :2] + dir2center * 0.05
    point2 = centers[:, :2] + dir2tan * 0.025 - dir2center * 0.05
    point3 = centers[:, :2] - dir2tan * 0.025 - dir2center * 0.05

    # Initialize figure
    fig = plt.figure(figsize=(12, 12))
    axis = fig.add_subplot(1, 1, 1)
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for p, (p1, p2, p3) in enumerate(zip(point1, point2, point3)):
        axis.plot([p1[0], p2[0]], [p1[1], p2[1]], c=colors[p % 6])
        axis.plot([p2[0], p3[0]], [p2[1], p3[1]], c=colors[p % 6])
        axis.plot([p3[0], p1[0]], [p3[1], p1[1]], c=colors[p % 6])

    if show_center:
        if centers.shape[1] == 2:
            axis.scatter(x, y, s=100, c='g')
        else:
            axis.scatter(x, y, s=100, c='g')
            for box in centers:
                rect = patches.Rectangle(
                    (box[0] - box[2] * 0.5, box[1] - box[3] * 0.5),
                    box[2], box[3],
                    linewidth=2, edgecolor='g', facecolor='none'
                )
                axis.add_patch(rect)
    if show_target:
        axis.scatter(targets[:, 0], targets[:, 1], s=100, c='r')
    if show_line:
        for x, y, ep in zip(centers[:, 0], centers[:, 1], end_point):
            axis.plot(
                [x, ep[0]], [y, ep[1]],
                c='b', linestyle='dashed'
            )
    if return_buf:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = np.array(Image.open(buf))  # H x W x 4
        image = image[:, :, :3]
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).unsqueeze(0)  # 1 x 3 x H x W
        plt.close('all')
        return image
    elif name is not None:
        plt.savefig(name)
        plt.close()
    return plt
