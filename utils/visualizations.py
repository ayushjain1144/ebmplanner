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


def plot_relations_2d(rel_boxes, ref_boxes=None, show=False):
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
            linewidth=2, edgecolor=colors[b % 5], facecolor='none'
        )
        axis.add_patch(rect)

    if ref_boxes is not None:
        for box in ref_boxes:
            if box.sum() == 0:
                continue
            rect = patches.Rectangle(
                (box[0] - box[2]*0.5, box[1] - box[3]*0.5), box[2], box[3],
                linewidth=2, edgecolor='b', facecolor='none'
            )
            axis.add_patch(rect)
    if show:
        plt.savefig(show)
        plt.show()
        return 0

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = np.array(Image.open(buf))  # H x W x 4
    image = image[:, :, :3]
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).unsqueeze(0)  # 1 x 3 x H x W
    plt.close('all')
    return image


def plot_triangles_2d(centers, angles, sizes):
    centers = centers[0]  # (B, N, 2) -> (N, 2)
    angles = angles[0]  # (N,)
    sizes = sizes[0]  # (N, 2)
    centers = centers[sizes.sum(1) > 0]
    angles = angles[sizes.sum(1) > 0]
    sizes = sizes[sizes.sum(1) > 0]
    # we need three points per triangle (who expected that...)
    dir2center = np.stack((np.cos(angles), np.sin(angles)), 1)
    dir2tan = np.stack((np.cos(angles + np.pi/2), np.sin(angles + np.pi/2)), 1)
    point1 = centers + dir2center * sizes[:, 0][:, None] / 2
    point2 = centers + dir2tan * sizes[:, 1][:, None] / 2
    point3 = centers - dir2tan * sizes[:, 1][:, None] / 2

    # Initialize figure
    fig = plt.figure(figsize=(12, 12))
    axis = fig.add_subplot(1, 1, 1)
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)
    colors = list(TABLEAU_COLORS.keys())

    for p, (p1, p2, p3) in enumerate(zip(point1, point2, point3)):
        axis.plot([p1[0], p2[0]], [p1[1], p2[1]], c=colors[p % 4])
        axis.plot([p2[0], p3[0]], [p2[1], p3[1]], c=colors[p % 4])
        axis.plot([p3[0], p1[0]], [p3[1], p1[1]], c=colors[p % 4])
    # plt.show()
    # return 0

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = np.array(Image.open(buf))  # H x W x 4
    image = image[:, :, :3]
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).unsqueeze(0)  # 1 x 3 x H x W
    plt.close('all')
    return image


def plot_lines_2d(points, centers, shape_centers=None, show=False):
    centers = centers[0]  # (B, N, 2) -> (N, 2)
    points = points[0]  # (N, 2)
    # centers = centers[(points.sum(1) > 0) & (centers.sum(1) > 0)]
    # points = points[(points.sum(1) > 0) & (centers.sum(1) > 0)]

    # Initialize figure
    fig = plt.figure(figsize=(12, 12))
    axis = fig.add_subplot(1, 1, 1)
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)
    colors = list(TABLEAU_COLORS.keys())

    for p, (p1, p2) in enumerate(zip(points, centers)):
        axis.plot([p1[0], p2[0]], [p1[1], p2[1]], c=colors[p % 4])
        axis.plot(p2[0], p2[1], marker='o', markersize=3, color=colors[p % 4])
    if shape_centers is not None:
        shape_centers = shape_centers[0]
        axis.plot(
            shape_centers[0], shape_centers[1],
            marker='x', markersize=3, color='r'
        )
        for p1 in centers:
            axis.plot(
                [p1[0], shape_centers[0]],
                [p1[1], shape_centers[1]],
                'g--'
            )
    if show:
        plt.show()
        return 0

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = np.array(Image.open(buf))  # H x W x 4
    image = image[:, :, :3]
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).unsqueeze(0)  # 1 x 3 x H x W
    plt.close('all')
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
