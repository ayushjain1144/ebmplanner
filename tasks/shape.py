"""Packing Box Pairs task."""

import os

import random
import numpy as np
from tasks.task import Task
import utils.transporter_utils as utils
import pybullet as p

import ipdb
st = ipdb.set_trace

class Shape(Task):
    """Shape Environment"""

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "rearrange {color} {objects} into a {shape}"

        self.task_completed_desc = "done rearranging objects."

        # circle's radius
        self.radius = {
            "large": 0.22,
            "medium": 0.17,
            "small": 0.13
        }

        self.pos = {
            "center": (0.5, 0.0),
            # "left side": (0.5, -0.25),
            # "right side": (0.5, 0.25),
        }

        self.objects = {
            "rings": "hanoi/disk3.urdf",
            "cubes": "box/box-template.urdf",
            "cylinders": "box/cylinder-template.urdf",
            "bowls": "bowl/bowl.urdf"
        }

        self.object_sizes = {
            "rings": (0.04, 0.04, 0.02),
            "cubes": (0.04, 0.08, 0.02),
            "cylinders": (0.04, 0.08, 0.02),
            "bowls": (0.04, 0.08, 0.02),
        }

        self.shape = None
        self.num_targets = None 
        self.save = True # saves gt labels
        self.eval_type = None

    def reset(self, env):
        super().reset(env)
        self.labels = {}

        # all colors + no color at all
        all_colors = self.get_colors()

        # select color
        target_color = random.choice(all_colors)

        # select object
        target_object = random.choice(list(self.objects.keys()))

        # select size
        target_size = random.choice(list(self.radius.keys()))

        # select 
        target_pos = random.choice(list(self.pos.keys()))

        utterance = self.lang_template.format(
            color=target_color if target_color != "none" else "all",
            objects=target_object,
            shape=self.shape
        )

        # print(utterance)

        # number of target objects
        if self.num_targets is None:
            num_targets = np.random.randint(5, 7)
        else:
            num_targets = random.choice(
            np.arange(self.num_targets[0], self.num_targets[1]+1)
        )

        # get goal locations
        locs = self.get_shape(num_targets,
                self.pos[target_pos], self.radius[target_size])
        
        # load dummy objects
        dummy_obj_ids = self.load_objects(
            env, 'bowls', num_targets, locs=locs
        )

        # load targets
        obj_ids = self.load_objects(
            env, target_object,
            num_targets, target_color)

        # spawn distractors
        
        # number of distractors
        num_distractors = np.random.randint(1, 2)
        distractor_color = "none" # when none, color gets selected randomly
        
        # remove target object from possible distractor objects
        all_distractor_objects = list(self.objects.keys())
        all_distractor_objects.remove(target_object)
        
        distractor_objects = random.choices(all_distractor_objects, k=num_distractors)
        # print(f"distractor objects: {distractor_objects}")
        dist_obj_ids = self.load_objects(
            env, distractor_objects, num_distractors, 
            distractor_color, distractor=True
        )

        # delete dummy objects
        self.delete_objects(dummy_obj_ids)
        
        # set goal locations
        goal_poses = self.set_goals(obj_ids, locs)

        # set task goals now really
        self.goals.append((
            obj_ids, np.eye(len(obj_ids)), goal_poses,
            False, True, self.eval_type, None, 1
        ))
        self.lang_goals.append(utterance)

        if self.save:
            return self.labels
    
    def load_objects(
        self, env, target_objects,
            num_targets, target_color="none", locs=None, distractor=False):
        
        # we are just adding some objects to avoid collisions later
        dummy = locs is not None

        obj_ids = []

        if isinstance(target_objects, str):
            target_objects = [target_objects] * num_targets
        
        assert len(target_objects) == num_targets

        if target_color == "none":
            colors = random.choices(self.get_colors(), k=num_targets)

        for i in range(num_targets):
            object_urdf = self.objects[target_objects[i]]
            size = self.object_sizes[target_objects[i]]
            obj_pose = self.get_random_pose(env, size)
            if obj_pose[0] == None:
                st()
                self.get_random_pose(env, size)
                assert False

            if target_objects[i] not in  ['bowls','rings'] :
                object_urdf = self.fill_template(object_urdf, {'DIM': (size[0],size[0],size[2])})
            object_id = env.add_object(object_urdf, obj_pose, dummy=dummy)
            obj_ids.append((object_id, (0, None)))
            # sentence doesn't mention color
            if target_color == "none":
                color = colors[i]
                # print(f"None color replaced by {color}")
            else:
                color = target_color
            p.changeVisualShape(object_id, -1, rgbaColor=utils.COLORS[color] + [1])
            
            # saving ground truths
            if not dummy:
                if not distractor:
                    env.obj_ids['move'].append(object_id)
                self.labels[object_id] = f"{target_color} {target_objects[i]}" \
                                        if target_color != "none" \
                                        else f"{target_objects[i]}"
        return obj_ids


    def delete_objects(self, dummy_obj_ids):
        for obj_id in dummy_obj_ids:
            p.removeBody(obj_id[0])
        return

    def set_goals(self, obj_ids, locs):
        assert len(obj_ids) == len(locs[0])

        goal_poses = []
        for obj_id, x, y in zip(obj_ids, locs[0], locs[1]):
            pose = p.getBasePositionAndOrientation(obj_id[0])
            goal_pose = (x, y, pose[0][2])
            goal_poses.append((goal_pose, pose[1]))
        return goal_poses

    def get_colors(self):
        raise NotImplementedError
    
    # implement this for every shape
    def get_shape(self):
        raise NotImplementedError

    def get_box_from_obj_id(self, obj_id, obj_mask):
        obj_loc = np.where(obj_mask == obj_id)
        x1, x2 = np.min(obj_loc[0]), np.max(obj_loc[0])
        y1, y2 = np.min(obj_loc[1]), np.max(obj_loc[1])
        return [x1, y1, x2, y2]


class CircleSeenColors(Shape):
    def __init__(self):
        super().__init__()
        self.shape = "circle"
        self.num_targets = [5, 7]
        self.eval_type = "polygon"

    def get_colors(self):
        return utils.TRAIN_COLORS

    def get_shape(self, num_objects, center, radius):
        xc, yc = center
        new_xs = xc + np.cos(2 * np.pi * np.arange(num_objects) / num_objects) * radius
        new_ys = yc + np.sin(2 * np.pi * np.arange(num_objects) / num_objects) * radius
        return new_xs, new_ys


class CircleUnseenColors(CircleSeenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS 


class CircleUnseenColors(CircleSeenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS 


class TriangleSeenColors(CircleSeenColors):
    def __init__(self):
        super().__init__()
        self.shape = "triangle"
        self.num_targets = [3, 3]

class TriangleUnseenColors(CircleUnseenColors):
    def __init__(self):
        super().__init__()
        self.shape = "triangle"
        self.num_targets = [3, 3]


class SquareSeenColors(CircleSeenColors):
    def __init__(self):
        super().__init__()
        self.shape = "square"
        self.num_targets = [4, 4]

class SquareUnseenColors(CircleUnseenColors):
    def __init__(self):
        super().__init__()
        self.shape = "square"
        self.num_targets = [4, 4]


class LineSeenColors(Shape):
    def __init__(self):
        super().__init__()
        self.shape = "line"
        self.num_targets = [3, 6]
        self.eval_type = "line"

        # line's distance between two points
        self.radius = {
            "large": 0.17,
            "medium": 0.13,
            "small": 0.09
        }

        self.pos = {
            # "top": (0.375, 0.0),
            "center": (0.5, 0.0),
            # "bottom": (0.625, 0.0),
        }

    def get_colors(self):
        return utils.TRAIN_COLORS

    def get_shape(self, num_objects, center, radius):
        xc, yc = center
        new_ys = np.arange(num_objects) * radius
        new_ys = (new_ys - new_ys.mean()) + yc
        new_xs = np.array([xc] * num_objects)
        return new_xs, new_ys


class LineUnseenColors(LineSeenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS