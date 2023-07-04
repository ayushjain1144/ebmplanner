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

class MultiCompositionalRelationsSeenColors(Task):
    """Shape Environment"""

    def __init__(self, depth=[3, 5]):
        super().__init__()
        self.depth = depth
        self.max_steps = 6

        # rel is 'on' 'the 'left 'of' | 'on' 'the' 'right' 'of' | 'above' | 'below'
        self.lang_template_rel = "put the {rel_color} {rel_object} to the {rel} of the {ref_color} {ref_object}"

        self.lang_template_video =  "put the {rel_color} {rel_object} {rel} {ref_color} {ref_object}"

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

        self.relation = None
        self.save = True # saves gt labels
        self.constraint_image = None

    def reset(self, env):
        super().reset(env)
        self.labels = {}

        all_objects = [f"{color} {object_}" for color in self.get_colors() for object_ in list(self.objects.keys())]

        depth = random.choice(
            np.arange(self.depth[0], self.depth[1]+1)
        )

        # select object
        objects_with_color = random.sample(all_objects, depth)
        all_objects = [obj for obj in all_objects if obj not in objects_with_color]

        rel_obj_names = []
        rel_obj_colors = []
        for obj in objects_with_color:
            color, obj_ = obj.split(" ")
            rel_obj_names.append(obj_.strip())
            rel_obj_colors.append(color.strip())

        # load objects as dummy
        rel_obj_ids, rel_obj_poses = self.load_objects(
            env, rel_obj_names, len(rel_obj_names), rel_obj_colors,
            dummy=True
        )

        rel_obj_xs = [rel_obj_poses[i][0][0] for i in range(len(rel_obj_poses))]
        rel_obj_ys = [rel_obj_poses[i][0][1] for i in range(len(rel_obj_poses))]

        # constraints
        constraint = 1 - np.eye(len(rel_obj_ids))
        rel_ids = []
        ref_ids = []
        relations = []

        i = 0
        _, _, obj_mask = self.get_true_image(env)

        while i < len(rel_obj_ids):
            # print(i)
            rel_obj_id = rel_obj_ids[i]
            ref_id_constraint = np.where(constraint[i])[0]
            if len(ref_id_constraint) == 0:
                env.set_failed_dategen(True)
                return None
            
            try:
                ref_obj_idx = random.choice(ref_id_constraint)
                ref_obj_id = rel_obj_ids[ref_obj_idx]
                constraint[ref_obj_idx, i] = 0
                relation_constraints = self.get_binary_relations(
                    rel_obj_id[0], ref_obj_id[0], obj_mask)
            # sometimes another object can completely mask the first object (not sure why)
            except Exception as e:
                env.set_failed_dategen(True)
                return None

            if len(relation_constraints) == 0:
                constraint[i, ref_obj_idx] = 0
                continue

            relation = random.choice(relation_constraints)
            rel_ids.append(i)
            ref_ids.append(ref_obj_idx)
            relations.append(relation)
            i += 1

        # make an utterance
        utterance = ""
        utterance_video = ""
        for i in range(len(rel_obj_ids)):
            utterance += self.lang_template_rel.format(
                rel_color = rel_obj_colors[rel_ids[i]],
                rel_object = rel_obj_names[rel_ids[i]],
                rel = relations[i],
                ref_color = rel_obj_colors[ref_ids[i]],
                ref_object = rel_obj_names[ref_ids[i]]
            )
            utterance += " and "

            utterance_video += self.lang_template_video.format(
                rel_color = rel_obj_colors[rel_ids[i]],
                rel_object = rel_obj_names[rel_ids[i]],
                rel = f"{relations[i]}" if relations[i] in ["above", "below"] else f"{relations[i]} of",
                ref_color = rel_obj_colors[ref_ids[i]],
                ref_object = rel_obj_names[ref_ids[i]]
            )
            utterance_video += " and "
        
        # remove trailing " and "
        utterance = utterance[:-5]
        utterance_video = utterance_video[:-5]
        print(utterance)
        

        # spawn distractors
        # number of distractors
        num_distractors = np.random.randint(1, 2)
                
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

        # spawn rel objects actually
        rel_ids_done = []
        rel_obj_ids_real = []
        for i in range(len(rel_obj_ids)):
            rel_id = rel_obj_ids[i][0]
            ref_id = rel_obj_ids[ref_ids[i]][0]
            if rel_id in rel_ids_done:
                constraint = True
            else:
                constraint = False

            rel_obj_id, ref_obj_pose = self.load_objects(
                env, rel_obj_names[i],
                1, [rel_obj_colors[i]], motion="move", 
                constraint=constraint, ref_obj_id=rel_id,
                weak=True, ref_relations=relations[i]
            )

            # can fail if constraints make it impossible :/
            if env.failed_datagen:
                return None
                
            rel_obj_ids_real.append(rel_obj_id[0])
            rel_ids_done.append(ref_id)

        # delete dummy objects
        self.delete_objects(rel_obj_ids)

        goal_check_info = {
            "rel_idxs": rel_ids,
            "ref_idxs": ref_ids,
            "relations": relations,
            "obj_ids": [obj_id[0] for obj_id in rel_obj_ids_real]
        }
        
        # # set goal locations
        goal_poses = self.set_goals(
            rel_obj_ids_real,
            [rel_obj_xs, rel_obj_ys],
            goal_check_info
        )

        # set task goals now really
        self.goals.append((
            rel_obj_ids_real, np.eye(len(rel_obj_ids_real)), goal_poses,
            False, True, 'multi_relation', None, 1
        ))
        self.lang_goals.append(utterance)
        self.lang_video_goals.append(utterance_video)

        if self.save:
            return self.labels
    
    def load_objects(
        self, env, target_objects,
            num_targets, target_colors="none", locs=None, distractor=False, 
            constraint=None, ref_obj_id=None, motion=None, dummy=False, 
            conjugate=False, ref_relations=None, weak=False):
        
        obj_ids = []
        obj_poses = []
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
                ref_relations=ref_relations, weak=weak
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
            obj_poses.append(obj_pose)

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

    def get_colors(self):
        return utils.TRAIN_COLORS

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

    def get_constraint(self, obj_mask, ref_obj_ids, ref_relations=None, conjugate=False):
        # if conjugate:
        #     # return flipped version of constrained image
        #     return (-self.constraint_image + 1)

        assert ref_relations is not None

        constraint_image_ = self.relation_to_class[ref_relations].get_constraint(obj_mask, ref_obj_ids)
        
        # Note: we want A to the left of B, we feed A as ref_obj_id
        # left as relation, and constraint_image_ is all empty area 
        # to the left of A, so we invert it to get all area to the right
        # of A where B can be spawned
        constraint_image_ = -constraint_image_ + 1

        # also set the constraint image in the self
        # for goals and conjugate 
        # self.constraint_image = constraint_image_

        return constraint_image_

class MultiCompositionalRelationsUnseenColors(MultiCompositionalRelationsSeenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS
