"""Sorting Task."""

import numpy as np
from tasks.task import Task
import utils.transporter_utils as utils

import pybullet as p


class PlaceRedInGreen(Task):
    """Sorting Task."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.block_color = 'red'
        self.bowl_color = 'green'
        self.lang_template = "put the red blocks in a green bowl"
        self.task_completed_desc = "done placing blocks in bowls."

    def reset(self, env):
        super().reset(env)
        n_bowls = np.random.randint(1, 4)
        n_blocks = np.random.randint(1, n_bowls + 1)
        labels = {}

        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_poses = []
        for _ in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size)
            box_id = env.add_object(bowl_urdf, bowl_pose, 'fixed')
            labels[box_id] = f'{self.bowl_color} bowl'
            bowl_poses.append(bowl_pose)
            color = utils.COLORS[self.bowl_color]
            if not box_id:
                continue
            p.changeVisualShape(box_id, -1, rgbaColor=color + [1])

        # Add blocks.
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for _ in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            labels[block_id] = f'{self.block_color} blocks'
            blocks.append((block_id, (0, None)))
            env.obj_ids['move'].append(block_id)
            color = utils.COLORS[self.block_color]
            if not block_id:
                continue
            p.changeVisualShape(block_id, -1, rgbaColor=color + [1])

        # Goal: each block is in a different bowl.
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
        self.lang_goals.append(self.lang_template)

        # Colors of distractor objects.
        bowl_colors = [c for c in utils.COLORS if c != self.bowl_color]
        block_colors = [c for c in utils.COLORS if c != self.block_color]
        # Add distractors.
        n_distractors = 0
        while n_distractors < 6:
            is_block = np.random.rand() > 0.5
            urdf = block_urdf if is_block else bowl_urdf
            size = block_size if is_block else bowl_size
            colors = block_colors if is_block else bowl_colors
            pose = self.get_random_pose(env, size)
            if not any(pose):  # in case it doesn't fit
                continue
            obj_id = env.add_object(urdf, pose)
            color = utils.COLORS[colors[n_distractors % len(colors)]]
            if not obj_id:
                continue
            labels[obj_id] = (
                colors[n_distractors % len(colors)] + 'block' if is_block
                else 'bowl'
            )
            p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
            n_distractors += 1
        return labels
