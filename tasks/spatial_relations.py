"""Packing Box Pairs task."""

import os
from copy import copy

import random
import numpy as np
from global_vars import IN_SHAPE, PIXEL_SIZE
from tasks.task import Task
import utils.transporter_utils as utils
import pybullet as p

import ipdb
st = ipdb.set_trace

class SpatialRelations(Task):
    """Shape Environment"""

    def __init__(self):
        super().__init__()
        self.max_steps = 2
        # rel is 'on' 'the 'left 'of' | 'on' 'the' 'right' 'of' | 'above' | 'below'
        self.lang_template = "put the {rel_color} {rel_object} to the {rel} of the {ref_color} {ref_object}"

        self.task_completed_desc = "done placing objects."

        self.objects = {
            "ring": "hanoi/disk3.urdf",
            "cube": "box/box-template.urdf",
            "cylinder": "box/cylinder-template.urdf",
            "bowl": "bowl/bowl.urdf"
        }

        self.object_sizes = {
            "ring": (0.04, 0.04, 0.02),
            "cube": (0.04, 0.08, 0.02),
            "cylinder": (0.04, 0.08, 0.02),
            "bowl": (0.04, 0.08, 0.02),
        }
        self.buffer = round((0.10 / PIXEL_SIZE) + 1)

        self.relation = None
        self.save = True # saves gt labels
        self.constraint_image = None

    def reset(self, env):
        super().reset(env)
        self.labels = {}

        # all colors + no color at all
        all_colors = copy(self.get_colors())

        # select color
        rel_color = random.choice(all_colors)

        # select object
        rel_object = random.choice(list(self.objects.keys()))

        all_colors.remove(rel_color)
        ref_color = random.choice(all_colors)
        ref_object = random.choice(list(self.objects.keys()))


        utterance = self.lang_template.format(
            rel_color=rel_color,
            rel_object=rel_object, rel=self.relation,
            ref_color=ref_color, ref_object=ref_object
        )

        print(utterance)

        # load ref object
        ref_obj_id, ref_obj_pose = self.load_objects(
            env, ref_object,
            1, ref_color, motion="fixed")
        
        # load dummy object in place of rel object's final location
        dummy_obj_ids, rel_gt_pose = self.load_objects(
            env, 'bowl', 1, constraint=True, ref_obj_id=ref_obj_id[0][0],
            dummy=True
        )

        # set constraint image
        rel_constraint_image = self.constraint_image

        # load only rel objects (since ref objects don't move)
        # conjugate ensures that the initial object configuration
        # is invalid
        rel_obj_id, rel_obj_pose = self.load_objects(
            env, rel_object,
            1, rel_color, constraint=True, motion="move", 
            conjugate=True, ref_obj_id=ref_obj_id[0][0])

        if env.failed_datagen:
            return None

        # spawn distractors
        
        # number of distractors
        num_distractors = np.random.randint(2, 5)
        distractor_color = "none" # when none, color gets selected randomly
        
        # remove target object from possible distractor objects
        all_distractor_objects = list(self.objects.keys())
        all_distractor_objects.remove(rel_object)
        if ref_object in all_distractor_objects:
            all_distractor_objects.remove(ref_object)
        
        distractor_objects = random.choices(all_distractor_objects, k=num_distractors)
        # print(f"distractor objects: {distractor_objects}")
        dist_obj_ids = self.load_objects(
            env, distractor_objects, num_distractors, 
            distractor_color, distractor=True
        )

        # delete dummy objects
        self.delete_objects(dummy_obj_ids)
        
        goal_check_info = {
            "rel_idxs": [0],
            'ref_idxs': [1],
            "relations": [self.relation],
            "obj_ids": [rel_obj_id[0][0], ref_obj_id[0][0]],
        }
        goal_poses = self.set_goals(
            rel_obj_id,
            [[rel_gt_pose[0][0][0]], [rel_gt_pose[0][0][1]]],
            goal_check_info
        )

        # set task goals now really
        self.goals.append((
            rel_obj_id, np.eye(len(rel_obj_id)), goal_poses,
            False, True, 'multi_relation', None, 1
        ))
        self.lang_goals.append(utterance)

        if self.save:
            return self.labels
    
    def load_objects(
        self, env, target_objects,
            num_targets, target_color="none", locs=None, distractor=False, 
            constraint=None, ref_obj_id=None, motion=None, dummy=False, 
            conjugate=False):
        
        obj_ids = []
        obj_poses = []
        constraint_fn = self.get_constraint if constraint else None
        if isinstance(target_objects, str):
            target_objects = [target_objects] * num_targets
        
        assert len(target_objects) == num_targets

        if target_color == "none":
            colors = random.choices(self.get_colors(), k=num_targets)

        for i in range(num_targets):
            object_urdf = self.objects[target_objects[i]]
            size = self.object_sizes[target_objects[i]]
            obj_pose = self.get_random_pose(
                env, size, constraint_fn=constraint_fn,
                ref_obj_id=ref_obj_id, conjugate=conjugate
            )
            if obj_pose[0] == None:
                print("Not Enough Space: Need to Resample")
                env.set_failed_dategen(True)
                return None, None

            if target_objects[i] not in  ['bowl','ring'] :
                object_urdf = self.fill_template(object_urdf, {'DIM': (size[0],size[0],size[2])})
            object_id = env.add_object(object_urdf, obj_pose, dummy=dummy)
            obj_ids.append((object_id, (0, None)))
            obj_poses.append(obj_pose)
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
                    env.obj_ids[motion].append(object_id)
                self.labels[object_id] = f"{target_color} {target_objects[i]}" \
                                        if target_color != "none" \
                                        else f"{target_objects[i]}"
        return obj_ids, obj_poses

    def delete_objects(self, dummy_obj_ids):
        for obj_id in dummy_obj_ids:
            p.removeBody(obj_id[0])
        return

    def set_goals(self, obj_ids, locs, goal_check_info):
        """
        Args:
            obj_ids (_type_): N
            locs (_type_): [[xs], [ys]] len(xs)=N

        Returns:
            _type_: _description_
        """
        assert len(obj_ids) == len(locs[0])

        goal_poses = []
        for obj_id, x, y in zip(obj_ids, locs[0], locs[1]):
            pose = p.getBasePositionAndOrientation(obj_id[0])
            goal_pose = (x, y, pose[0][2])
            goal_poses.append((goal_pose, pose[1], goal_check_info))
        return goal_poses

    def get_colors(self):
        raise NotImplementedError

    def get_constraint(self, obj_mask, ref_obj_id, conjugate=False, ref_relations=None):
        constraint_image = np.zeros_like(obj_mask)
        ref_obj_loc = np.where(obj_mask == ref_obj_id)
        x1, x2 = np.min(ref_obj_loc[0]), np.max(ref_obj_loc[0])
        y1, y2 = np.min(ref_obj_loc[1]), np.max(ref_obj_loc[1])

        if conjugate:
            # return flipped version of constrained image
            return (-self.constraint_image + 1)

        self.constraint_image = self.get_relation(x1, y1, x2, y2, constraint_image)
        return self.constraint_image


    def get_binary_relations(self, rel_obj_id, ref_obj_id, obj_mask):
        """Gets all pairs of relations between all object ids"""

        # get their boxes
        rel_box = self.get_box_from_obj_id(rel_obj_id, obj_mask)
        ref_box = self.get_box_from_obj_id(ref_obj_id, obj_mask)

        relations_ = []

        x1, y1, x2, y2 = rel_box
        x1_, y1_, x2_, y2_ = ref_box

        if x2 < x1_:
            relations_.append("left")
        if x1 > x2_:
            relations_.append("right")
        if y2 < y1_:
            relations_.append("above")
        if y1 > y2_:
            relations_.append("below")
        
        return relations_


    def get_box_from_obj_id(self, obj_id, obj_mask):
        obj_loc = np.where(obj_mask == obj_id)
        x1, x2 = np.min(obj_loc[0]), np.max(obj_loc[0])
        y1, y2 = np.min(obj_loc[1]), np.max(obj_loc[1])
        return [x1, y1, x2, y2]

    # implement this for every shape
    def get_relation(self):
        raise NotImplementedError


class LeftSeenColors(SpatialRelations):
    def __init__(self):
        super().__init__()
        self.relation = "left"

    def get_colors(self):
        return utils.TRAIN_COLORS

    def get_relation(self, x1, y1, x2, y2, constraint_image):
        if x1 - self.buffer < 0:
            return constraint_image
        constraint_image[:x1-self.buffer, :] = 1
        return constraint_image

class LeftUnseenColors(LeftSeenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS

class RightSeenColors(SpatialRelations):
    def __init__(self):
        super().__init__()
        self.relation = "right"

    def get_colors(self):
        return utils.TRAIN_COLORS

    def get_relation(self, x1, y1, x2, y2, constraint_image):
        if x2 + self.buffer >= constraint_image.shape[0]:
            return constraint_image
        constraint_image[x2+self.buffer:, :] = 1
        return constraint_image


class RightUnseenColors(RightSeenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class AboveSeenColors(SpatialRelations):
    def __init__(self):
        super().__init__()
        self.relation = "above"

    def get_colors(self):
        return utils.TRAIN_COLORS

    def get_relation(self, x1, y1, x2, y2, constraint_image):
        if y1 - self.buffer < 0:
            return constraint_image
        constraint_image[:, :y1-self.buffer] = 1
        return constraint_image


class AboveUnseenColors(AboveSeenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class BelowSeenColors(SpatialRelations):
    def __init__(self):
        super().__init__()
        self.relation = "below"

    def get_colors(self):
        return utils.TRAIN_COLORS

    def get_relation(self, x1, y1, x2, y2, constraint_image):
        if y2 + self.buffer >= constraint_image.shape[1]:
            return constraint_image
        constraint_image[:, y2+self.buffer:] = 1
        return constraint_image

class BelowUnseenColors(BelowSeenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS
