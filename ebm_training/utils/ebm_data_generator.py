import json
import math

import numpy as np
from shapely.geometry import MultiPoint
from tqdm import tqdm

XMIN = -0.5
XMAX = 0.5
YMIN = -1
YMAX = 1
ZMIN = -1
ZMAX = 1
SIZE = 0.15


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
            'stack': self.sample_nboxes_stack,
            'pizza': self.sample_ntriangles_pizza,
            'racer': self.sample_ntriangles_racer,
            'table': self.sample_table,
            'alpha': self.sample_alpha
        }
        self.lens = np.array(
            [XMAX - XMIN - SIZE, YMAX - YMIN - SIZE, ZMAX - ZMIN - SIZE]
        ) / 2.0

    def sample(self, nsamples, labels=None):
        """Sample a given number of positives for each class."""
        if not labels:
            labels = self.executors.keys()
        samples = []
        for label in labels:
            print('Data for ' + label)
            label_samples = [
                self.sample_one(label) for _ in tqdm(range(nsamples))
            ]
            samples += list(label_samples)
        return samples

    def sample_one(self, label):
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
                label, np.random.randint(3, 7), 0
            )
        if label in ('circle', 'alpha'):
            return self._sample_shape(
                label,
                np.random.randint(5, 8), 0
                # np.random.randint(3, 9), 0
            )
        if label == 'center':
            return self._sample_shape(
                label, 4,  # np.random.randint(3, 12),
                1
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
        if label == 'pizza':
            return self.sample_ntriangles_pizza(np.random.randint(6, 12))
        if label == 'racer':
            return self.sample_ntriangles_racer(6)
        if label == 'table':
            return self.sample_table()
        if label == 'alpha':
            return self.sample_alpha()
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
        # Attention
        attention = np.ones(len(all_ref_boxes))
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

    def sample_table(self):
        """Make a table example."""
        table_edge = max(XMIN, YMIN)

        # Sample plate box in front of edge
        edge_box = np.array([0, table_edge + 0.05, 0, 0.1, 0.1, 0.1])
        plate_box = self._sample_given_ref('behind', edge_box)[0]
        plate_box[1] = edge_box[1] + 0.05
        m = max(plate_box[3], plate_box[4])
        plate_box[3] = m
        plate_box[4] = m

        # Sample napkin right of plate
        napkin_box = self._sample_given_ref('right', plate_box)[0]
        napkin_box[3:] = plate_box[3:] * 0.5
        napkin_box[1] = plate_box[1]

        # Sample fork on napkin
        fork_box = self._sample_given_ref('above', napkin_box)[0]
        fork_box[3] = min(fork_box[3], fork_box[4] / 8)
        fork_box[:3] = napkin_box[:3]

        # Sample knife left of plate
        knife_box = self._sample_given_ref('left', plate_box)[0]
        knife_box[3:] = fork_box[3:]
        knife_box[1] = fork_box[1]
        knife_box[0] = 2 * plate_box[0] - fork_box[0]

        # Sample bowl in front of plate
        bowl_box = self._sample_given_ref('behind', plate_box)[0]
        m = max(bowl_box[3], bowl_box[4])
        bowl_box[3] = m
        bowl_box[4] = m
        bowl_box[1] = min(bowl_box[1], min(XMAX, YMAX) - bowl_box[4])
        return {
            'edge': table_edge,
            'plate': plate_box.tolist(),
            'napkin': napkin_box.tolist(),
            'fork': fork_box.tolist(),
            'knife': knife_box.tolist(),
            'bowl': bowl_box.tolist(),
            'label': 'table',
            'angles': [np.pi / 2, np.pi / 2, np.pi / 2]
        }

    def _sample_given_ref(self, label, ref):
        rel = []
        ref = self._to_min_max(ref[None, :]).flatten()
        while not rel:
            new_box = self.sample_nboxes(1, [])
            mmbox = self._to_min_max(new_box).flatten()
            if self.executors[label](mmbox, ref):
                rel.append(np.copy(new_box.flatten()))
        return rel

    def sample_nboxes(self, n_points, old_boxes=None, constrain_z=[]):
        """Sample n boxes in the space (cube [-0.85, 0.85])."""
        boxes = []
        if old_boxes is not None:
            boxes += [np.asarray(box) for box in old_boxes]
        else:
            old_boxes = []
        while len(boxes) < n_points + len(old_boxes):
            new_box = np.concatenate((
                2 * self.lens * np.random.rand(3) - self.lens,  # center
                0.2 * np.random.rand(3) + 0.1  # size
            ))
            if constrain_z:
                new_box[2] = constrain_z[np.random.randint(len(constrain_z))]
            if not any(self._intersect(new_box, box) for box in boxes):
                boxes.append(np.copy(new_box))
        return np.stack(boxes)[-n_points:]

    def sample_nboxes_line_(self, n_points):
        """Sample n boxes in the space forming a line."""
        # Define a line
        x1, y1, z1 = 2 * self.lens * np.random.rand(3) - self.lens
        x2, y2 = 2 * self.lens[:2] * np.random.rand(2) - self.lens[:2]
        while np.abs(x2-x1)/n_points < 0.1 or np.abs(y2-y1)/n_points < 0.1:
            x1, y1, z1 = 2 * self.lens * np.random.rand(3) - self.lens
            x2, y2 = 2 * self.lens[:2] * np.random.rand(2) - self.lens[:2]
        z1 = 0
        line = [(x1, y1, z1), (x2, y2, z1)]
        # Sample boxes
        boxes = []
        print(n_points, np.abs(x2-x1), np.abs(y2-y1))
        while len(boxes) < n_points:
            # Random box
            new_box = np.concatenate((
                2 * self.lens * np.random.rand(3) - self.lens,  # center
                (0.1 * np.random.rand(3) + 0.1) / n_points  # size
            ))
            new_box[0] = len(boxes) * (x2 - x1) / n_points + x1
            # Condition random y, z to x to satify line concept
            new_box[1], new_box[2] = self.get_yz_on_line(new_box[0], line)
            print(new_box)
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

    def sample_nboxes_line(self, n_points):
        """Sample n boxes in the space forming a line."""
        # Define a line
        x1, y1, z1 = 2 * self.lens * np.random.rand(3) - self.lens
        x2, y2 = 2 * self.lens[:2] * np.random.rand(2) - self.lens[:2]
        while np.abs(x2 - x1) / n_points < 0.1 or np.abs(y2 - y1) / n_points < 0.1:
            x1, y1, z1 = 2 * self.lens * np.random.rand(3) - self.lens
            x2, y2 = 2 * self.lens[:2] * np.random.rand(2) - self.lens[:2]
        # z1 = 0
        x2 = x1
        # line = [(x1, y1, z1), (x2, y2, z1)]
        # Sample boxes
        boxes = []
        while len(boxes) < n_points:
            # Random box
            new_box = np.concatenate((
                2 * self.lens * np.random.rand(3) - self.lens,  # center
                (0.1 * np.random.rand(3) + 0.1) / n_points  # size
            ))
            new_box[0] = x1
            new_box[1] = len(boxes) * (y2 - y1) / n_points + y1
            # Condition random y, z to x to satify line concept
            # new_box[1], new_box[2] = self.get_yz_on_line(new_box[0], line)
            new_box[2] = 0
            # Check if the new box intersects an existing
            if not any(self._intersect(new_box, box) for box in boxes):
                boxes.append(np.copy(new_box))
        return np.stack(boxes)

    def sample_nboxes_circle(self, n_points):
        """Sample n boxes in the space forming a circle."""
        # Sample center and radius
        while True:
            # center
            xc, yc = 2 * self.lens[:2] * np.random.rand(2) - self.lens[:2]

            # circle should fit in our space
            low = SIZE
            high = min(self.lens[:2] - [abs(xc), abs(yc)])

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
                    2 * self.lens * np.random.rand(3) - self.lens,  # center
                    0.5 * radius * ((SIZE - 0.075) * np.random.rand(3) + 0.075)  # size
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
            assert (new_box[:2] - new_box[3:4] / 2.0 >= np.array([XMIN, YMIN])).all(), new_box
            assert (new_box[:2] + new_box[3:4] / 2.0 <= np.array([XMAX, YMAX])).all()
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
        return self.sample_nboxes_circle(4)

    def sample_nboxes_triangle(self, n_points):
        """Sample n boxes in the space forming a triangle."""
        items = self.sample_nboxes_circle(3)
        return items

    def sample_nboxes_cube(self, n_points):
        """Sample n boxes in the space forming a cube."""
        # Sample center, range of a cude that can fit
        while True:
            # sample center
            xc, yc, zc = 1.7 * np.random.rand(3) - 0.85

            # length range
            low = SIZE
            high = 0.85 - max(abs(xc), abs(yc), abs(zc))

            if high < low:
                continue
            length = np.random.uniform(low=low, high=high)
            break
        pos = [
            (xc - length / 2.0, yc - length / 2.0, zc - length / 2.0),
            (xc - length / 2.0, yc - length / 2.0, zc + length / 2.0),
            (xc - length / 2.0, yc + length / 2.0, zc - length / 2.0),
            (xc + length / 2.0, yc - length / 2.0, zc - length / 2.0),
            (xc + length / 2.0, yc + length / 2.0, zc - length / 2.0),
            (xc + length / 2.0, yc - length / 2.0, zc + length / 2.0),
            (xc - length / 2.0, yc + length / 2.0, zc + length / 2.0),
            (xc + length / 2.0, yc + length / 2.0, zc + length / 2.0)
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
        l_new = np.random.uniform(low=0.01, high=0.9 * l_old)
        w_new = np.random.uniform(low=0.01, high=0.9 * w_old)
        h_new = np.random.uniform(low=0.01, high=0.9 * h_old)

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

    def sample_ntriangles_pizza(self, n_points):
        """Sample n triangles that face towards a center."""
        boxes = self.sample_nboxes_circle(n_points)
        # Estimate radius and center
        points = boxes[:, :3]
        dists = ((points[:, None] - points[None, :]) ** 2).sum(2)
        radius = np.sqrt(dists.max()) / 2  # max dist approximates diameter
        center = points.mean(0)
        # Sample a size for triangles
        main_axis_size = np.random.uniform(0, radius / 2)
        sec_axis_size = np.random.uniform(0, radius / 2)
        sizes = np.ones((len(points), 2))
        sizes[:, 0] *= main_axis_size
        sizes[:, 1] *= sec_axis_size
        # Directions
        directions = (
            (center[None, :] - points)
            / np.sqrt(((center[None, :] - points) ** 2).sum(1))[:, None]
        )[:, :2]
        angle = np.arccos(directions[:, 0]) * (-1)**(directions[:, 1] < 0)
        return {
            'points': points[:, :2].tolist(),
            'angles': angle.tolist(),
            'sizes': sizes.tolist(),
            'label': 'pizza'
        }

    def sample_ntriangles_racer(self, n_points):
        """Sample n triangles that face towards a center."""
        items = self.sample_ntriangles_pizza(n_points)
        items['angles'] = np.array(items['angles']) - np.pi / 2
        items['angles'] = items['angles'].tolist()
        items['label'] = 'racer'
        return items

    def sample_alpha(self, n_points=6):
        """Sample 6 points making a capital A."""
        # Sample top point
        xc, yc = 2 * self.lens[:2] * np.random.rand(2) - self.lens[:2]
        zc = 0
        # Sample mid point in [(yc-len[1]*0.5)/2, yc]
        dy = max(0.1, 0.5 * (yc + self.lens[1] * 0.5) * np.random.rand(1)[0])
        ym = yc - dy
        xm = xc
        zm = zc
        # Sample mid flyers
        mid_len = 0.5 * min((np.random.rand(1)[0] + 1) * (yc - ym), self.lens[1])
        xf1 = xm - mid_len * 0.5
        xf2 = xm + mid_len * 0.5
        yf1 = ym
        yf2 = ym
        zf1 = zc
        zf2 = zc
        # Sample bottom flyers
        xb1 = xf1 - mid_len * 0.5
        xb2 = xf2 + mid_len * 0.5
        yb1 = yf1 - (yc - ym)
        yb2 = yf2 - (yc - ym)
        zb1 = zc
        zb2 = zc
        # Merge points
        points = np.array([
            [xc, yc, zc],
            [xm, ym, zm],
            [xf1, yf1, zf1],
            [xf2, yf2, zf2],
            [xb1, yb1, zb1],
            [xb2, yb2, zb2]
        ])
        # Sample boxes
        boxes = []
        for i in range(len(points)):
            add_it = False
            while not add_it:
                # Random box
                new_box = np.concatenate((
                    2 * self.lens * np.random.rand(3) - self.lens,  # center
                    (yc - ym) * ((SIZE - 0.075) * np.random.rand(3) + 0.075)
                ))
                new_box[:3] = points[i]
                # Check if the new box intersects an existing
                add_it = not any(self._intersect(new_box, box) for box in boxes)
            boxes.append(np.copy(new_box))
        points = np.stack(boxes)
        # Move inside range
        points[:, 0] += max(0, -self.lens[0] / 2 - points[:, 0].min() + 0.1)
        points[:, 1] += max(0, -self.lens[1] / 2 - points[:, 1].min() + 0.1)
        points[:, 0] -= min(0, points[:, 0].max() - self.lens[0] / 2 + 0.1)
        points[:, 1] -= min(0, points[:, 1].max() - self.lens[1] / 2 + 0.1)
        return points

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
            > 0.2 * min(ref_box[3] - ref_box[0], box[3] - box[0])
        )

    @staticmethod
    def _same_y_range(box, ref_box):
        return (
            min(box[4], ref_box[4]) - max(box[1], ref_box[1])
            > 0.2 * min(ref_box[4] - ref_box[1], box[4] - box[1])
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
            # and self._same_y_range(box, ref_box)
            # and self._same_z_range(box, ref_box)
        )

    def _is_right(self, box, ref_box):
        return (
            box[0] > ref_box[3]  # x_min > x_ref_max
            # and self._same_y_range(box, ref_box)
            # and self._same_z_range(box, ref_box)
        )

    def _is_front(self, box, ref_box):
        return (
            box[4] < ref_box[1]  # y_max < y_ref_min
            # and self._same_x_range(box, ref_box)
            # and self._same_z_range(box, ref_box)
        )

    def _is_behind(self, box, ref_box):
        return (
            box[1] > ref_box[4]  # y_min > y_ref_max
            # and self._same_x_range(box, ref_box)
            # and self._same_z_range(box, ref_box)
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
            int2box_ratio > SIZE  # xy intersection
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


def create_graph_data(filename, concept='right', nsamples=48):
    """Create graph annotations."""
    ntrain = nsamples // 2
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
