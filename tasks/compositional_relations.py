"""Packing Box Pairs task."""

import os
from copy import copy

import random
import numpy as np
from global_vars import IN_SHAPE
from tasks.task import Task
import utils.transporter_utils as utils
import tasks.spatial_relations as spatial_relations
import pybullet as p

import ipdb
st = ipdb.set_trace

class CompositionalRelationsSeenColors(Task):
    """Shape Environment"""

    def __init__(self, depth=[2, 4]):
        super().__init__()
        self.depth = depth
        self.max_steps = 2

        # rel is 'on' 'the 'left 'of' | 'on' 'the' 'right' 'of' | 'above' | 'below'
        self.lang_template_rel = ""

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

        self.relation_to_class = {
            "left": spatial_relations.LeftSeenColors(),
            "right": spatial_relations.RightSeenColors(),
            "above": spatial_relations.AboveSeenColors(),
            "below": spatial_relations.BelowSeenColors()
        }

        self.conjugate = {
            "left": "right",
            "right": "left",
            "above": "below",
            "below": "above"
        }

        self.relation = None
        self.save = True # saves gt labels
        self.constraint_image = None

    def reset(self, env):
        super().reset(env)
        self.labels = {}

        all_objects = [f"{color} {object_}" for color in self.get_colors() for object_ in list(self.objects.keys())]

        # select object
        rel_object_with_color = random.choice(all_objects)
        all_objects.remove(rel_object_with_color)
        rel_color, rel_object = rel_object_with_color.split(" ")
        rel_object = rel_object.strip()
        rel_color = [rel_color.strip()]

        utterance = ""
        video_utterance = f"put the {rel_object_with_color} "

        depth = random.choice(
            np.arange(self.depth[0], self.depth[1]+1)
        )

        # sample ref objects
        ref_objects = []
        ref_colors = []
        ref_relations = []
        for i in range(depth):
            # sample ref object
            ref_object_with_color = random.choice(all_objects)
            ref_color, ref_object = ref_object_with_color.split(' ')

            all_objects.remove(ref_object_with_color)
            ref_objects.append(ref_object.strip())
            ref_colors.append(ref_color.strip())

            # sample relations
            ref_rel = random.choice(list(self.relation_to_class.keys()))
            ref_relations.append(ref_rel)

            utterance += f'put the {rel_object_with_color} to the {ref_rel} of the {ref_object_with_color} and '

            if ref_rel in ['above', 'below']:
                video_utterance += f"{ref_rel} {ref_object_with_color} and "
            else:
                video_utterance += f"{ref_rel} of {ref_object_with_color} and "
        
        # remove trailing " and "
        utterance = utterance[:-5]
        video_utterance = video_utterance[:-5]
        print(utterance)

        # load dummy object in place of rel object's final location
        # without any constraints
        dummy_obj_ids, rel_gt_pose = self.load_objects(
            env, 'bowl', 1, dummy=True
        )

        # load ref object based on the constraints (if rel object should be left
        # of ref object, we need ref object to be right of rel object)
        # conjugate takes care of that
        rel_constraint_image = np.ones((IN_SHAPE[0], IN_SHAPE[1]))

        ref_obj_ids = []
        for ref_object, ref_color, ref_relation in zip(ref_objects, ref_colors, ref_relations):
            ref_obj_id, ref_obj_pose = self.load_objects(
                env, ref_object,
                1, [ref_color], motion="fixed", 
                constraint=True, ref_obj_id=dummy_obj_ids[0][0],
                ref_relations=ref_relation)
                
            # can fail if constraints make it impossible :/
            if env.failed_datagen:
                return None

            # get constraint
            _, _, obj_mask = self.get_true_image(env)
            constraint_image = self.relation_to_class[ref_relation].get_constraint(obj_mask, ref_obj_id[0][0])
            rel_constraint_image *= constraint_image

            relations = self.get_binary_relations(dummy_obj_ids[0][0], ref_obj_id[0][0], obj_mask)
            if ref_relation not in relations:
                st()

            ref_obj_ids.append(ref_obj_id)

        self.constraint_image = rel_constraint_image

        # load only rel objects (since ref objects don't move)
        # conjugate ensures that the initial object configuration
        # is initially violating constraints)
        rel_obj_id, rel_obj_pose = self.load_objects(
            env, rel_object,
            1, rel_color, constraint=True, motion="move", 
            conjugate=True)

        if env.failed_datagen:
            return None

        # spawn distractors
        
        # number of distractors
        num_distractors = np.random.randint(1, 3)
                
        distractor_objects_with_color = random.choices(all_objects, k=num_distractors)
        dist_objects, dist_colors = [], []
        for obj in distractor_objects_with_color:
            color, obj_ = obj.split(" ")
            dist_objects.append(obj_.strip())
            dist_colors.append(color.strip())


        # print(f"distractor objects: {distractor_objects}")
        dist_obj_ids = self.load_objects(
            env, dist_objects, num_distractors, 
            dist_colors, distractor=True
        )

        # delete dummy objects
        self.delete_objects(dummy_obj_ids)
        all_obj_ids = [rel_obj_id[0][0]] + [ref_obj_id[0][0] for ref_obj_id in ref_obj_ids]

        goal_check_info = {
            "rel_idxs": [0] * len(ref_relations),
            "ref_idxs": np.arange(1, len(ref_obj_ids) + 1),
            "relations": ref_relations,
            "obj_ids": all_obj_ids
        }
        
        # # set goal locations
        goal_poses = self.set_goals(
            rel_obj_id,
            [[rel_gt_pose[0][0]], [rel_gt_pose[0][1]]],
            goal_check_info)

        # set task goals now really
        self.goals.append((
            rel_obj_id, np.eye(len(rel_obj_id)), goal_poses,
            False, True, 'multi_relation', None, 1
        ))
        self.lang_goals.append(utterance)
        self.lang_video_goals.append(video_utterance)

        if self.save:
            return self.labels
    
    def load_objects(
        self, env, target_objects,
            num_targets, target_colors="none", locs=None, distractor=False, 
            constraint=None, ref_obj_id=None, motion=None, dummy=False, 
            conjugate=False, ref_relations=None):
        
        obj_ids = []
        constraint_fn = self.get_constraint if constraint else None
        if isinstance(target_objects, str):
            target_objects = [target_objects] * num_targets
        
        assert len(target_objects) == num_targets

        if target_colors == "none":
            colors = random.choices(self.get_colors(), k=num_targets)

        for i in range(num_targets):
            object_urdf = self.objects[target_objects[i]]
            size = self.object_sizes[target_objects[i]]
            obj_pose = self.get_random_pose(
                env, size, constraint_fn=constraint_fn,
                ref_obj_id=ref_obj_id, conjugate=conjugate,
                ref_relations=ref_relations
            )
            if obj_pose[0] == None:
                print("Not Enough Space: Need to Resample")
                env.set_failed_dategen(True)
                return None, None

            if target_objects[i] not in  ['bowl','ring'] :

                # to be more conservative in reserving space
                if dummy:
                    size = [size[0]+0.02, size[1]+0.02, size[2]]

                object_urdf = self.fill_template(object_urdf, {'DIM': (size[0],size[0],size[2])})
            object_id = env.add_object(object_urdf, obj_pose, dummy=dummy)
            obj_ids.append((object_id, (0, None)))

            # sentence doesn't mention color
            if target_colors == "none":
                color = colors[i]
                # print(f"None color replaced by {color}")
            else:
                color = target_colors[i]
            p.changeVisualShape(object_id, -1, rgbaColor=utils.COLORS[color] + [1])
            
            # saving ground truths
            if not dummy:
                if not distractor:
                    env.obj_ids[motion].append(object_id)
                self.labels[object_id] = f"{target_colors[i]} {target_objects[i]}" \
                                        if target_colors != "none" \
                                        else f"{target_objects[i]}"
        return obj_ids, obj_pose

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

    def get_constraint(self, obj_mask, ref_obj_ids, ref_relations=None, conjugate=False):
        if conjugate:
            # return flipped version of constrained image
            return (-self.constraint_image + 1)

        assert ref_relations is not None

        constraint_image_ = self.relation_to_class[self.conjugate[ref_relations]].get_constraint(obj_mask, ref_obj_ids)
        
        # Note: we want A to the left of B, we feed A as ref_obj_id
        # left as relation, and constraint_image_ is all empty area 
        # to the left of A, so we invert it to get all area to the right
        # of A where B can be spawned
        # constraint_image_ = -constraint_image_ + 1

        # also set the constraint image in the self
        # for goals and conjugate 
        # self.constraint_image = constraint_image_

        return constraint_image_
    

    def get_box_from_obj_id(self, obj_id, obj_mask):
        obj_loc = np.where(obj_mask == obj_id)
        x1, x2 = np.min(obj_loc[0]), np.max(obj_loc[0])
        y1, y2 = np.min(obj_loc[1]), np.max(obj_loc[1])
        return [x1, y1, x2, y2]


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

    def get_colors(self):
        return utils.TRAIN_COLORS


class CompositionalRelationsUnSeenColors(CompositionalRelationsSeenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS
