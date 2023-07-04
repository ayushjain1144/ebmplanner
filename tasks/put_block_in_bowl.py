"""Put Blocks in Bowl Task."""

import numpy as np
from tasks.task import Task
import utils.transporter_utils as utils
import pickle
import os

import random
import pybullet as p

import ipdb
st = ipdb.set_trace


class PutBlockInBowlUnseenColors(Task):
    """Put Blocks in Bowl base class and task."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "put the {pick} blocks in a {place} bowl"
        self.task_completed_desc = "done placing blocks in bowls."
        self.cnt = 0
        self.save=False

    def reset(self, env):
        super().reset(env)

        n_bowls = np.random.randint(1, 4)
        n_blocks = np.random.randint(1, n_bowls + 1)

        all_color_names = self.get_colors()
        selected_color_names = random.sample(all_color_names, 2)
        colors = [utils.COLORS[cn] for cn in selected_color_names]

        # import ipdb;ipdb.set_trace()
        all_obj = []
        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_poses = []
        labels = {}
        for _ in range(n_bowls):
            # import ipdb;ipdb.set_trace()
            bowl_pose = self.get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, 'fixed')
            # print(bowl_id)
            p.changeVisualShape(bowl_id, -1, rgbaColor=colors[1] + [1])
            bowl_poses.append((bowl_pose[0],bowl_pose[1],bowl_size))
            all_obj.append((bowl_pose[0],bowl_pose[1],bowl_size))
            labels[bowl_id] = selected_color_names[1] +' bowl' 

        # Add blocks.
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        block_poses = []
        # print('*******************')
        for _ in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            # print(block_id)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[0] + [1])
            blocks.append((block_id, (0, None)))
            env.obj_ids['move'].append(block_id)
            block_poses.append((block_pose[0],block_pose[1],block_size))
            all_obj.append((block_pose[0],block_pose[1],block_size))
            labels[block_id] = selected_color_names[0] +' block'

        # Goal: put each block in a different bowl.
        self.goals.append((
            blocks,  # objects to move
            np.ones((len(blocks), len(bowl_poses))),  # matches
            bowl_poses,  # target poses
            False,  # replace
            True,  # rotations
            'pose',  # metric
            None,  # params
            1  # max reward
        ))
        self.lang_goals.append(self.lang_template.format(pick=selected_color_names[0],
                                                         place=selected_color_names[1]))

        # Only one mistake allowed.
        self.max_steps = len(blocks) + 1

        # Colors of distractor objects.
        # distractor_bowl_colors = [utils.COLORS[c] for c in utils.COLORS if c not in selected_color_names]
        # distractor_block_colors = [utils.COLORS[c] for c in utils.COLORS if c not in selected_color_names]
        # import ipdb;ipdb.set_trace()
        distractor_bowl_colors = [c for c in utils.COLORS if c not in selected_color_names]
        distractor_block_colors = [c for c in utils.COLORS if c not in selected_color_names]
        # Add distractors.
        n_distractors = 0
        max_distractors = 6
        # print('.................')
        while n_distractors < max_distractors:
            is_block = np.random.rand() > 0.5
            urdf = block_urdf if is_block else bowl_urdf
            size = block_size if is_block else bowl_size
            colors = distractor_block_colors if is_block else distractor_bowl_colors
            pose = self.get_random_pose(env, size)
            all_obj.append((pose[0],pose[1],size))
            if not pose:
                continue
            obj_id = env.add_object(urdf, pose,'deformable')
            # print(obj_id)
            color = colors[n_distractors % len(colors)]
            if not obj_id:
                continue
            p.changeVisualShape(obj_id, -1, rgbaColor=utils.COLORS[color] + [1])

            labels[obj_id] = color +' block' if is_block else color +' bowl'
            n_distractors += 1

        # file_object = open('sample.txt', 'a')
        # file_object.write(str(all_obj))
        # file_object.write('\n')
        # # file_object.write(str(bowl_poses))
        # # file_object.write('\n')
        # # file_object.write(str(block_poses))
        # # file_object.write('\n')
        # # file_object.write(str(all_obj))
        # # file_object.write('\n')
        # # Close the file
        # file_object.close()
        if self.save:
            outfile = open(os.getcwd()+'/language/{}.pickle'.format(str(self.cnt)),'wb')
            pickle.dump(labels,outfile)
            outfile.close()
        self.cnt += 1
        return labels
    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class PutBlockInBowlSeenColors(PutBlockInBowlUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        return utils.TRAIN_COLORS


class PutBlockInBowlFull(PutBlockInBowlUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        all_colors = list(set(utils.TRAIN_COLORS) | set(utils.EVAL_COLORS))
        return all_colors