"""Base Task class."""

import binascii
import collections
import os
import tempfile
import time

import cv2
import numpy as np

from global_vars import BOUNDS, PIXEL_SIZE, IN_SHAPE
import cameras
from tasks import primitives
from tasks.grippers import Suction
import utils.transporter_utils as utils

import pybullet as p

import ipdb
st = ipdb.set_trace


class Task:
    """Base Task class."""

    def __init__(self):
        self.ee = Suction
        self.mode = 'train'
        self.name = None
        self.sixdof = False
        self.primitive = primitives.PickPlace()
        self.oracle_cams = cameras.Oracle.CONFIG

        # Evaluation epsilons (for pose evaluation metric).
        self.pos_eps = 0.01
        self.rot_eps = np.deg2rad(15)

        # Workspace bounds.
        self.pix_size = PIXEL_SIZE
        self.bounds = BOUNDS
        self.zone_bounds = np.copy(self.bounds)

        self.goals = []
        self.lang_goals = []
        self.lang_video_goals = []
        self.task_completed_desc = "task completed."
        self.progress = 0
        self._rewards = 0
        self.assets_root = None


    def reset(self, env):  # pylint: disable=unused-argument
        if not self.assets_root:
            raise ValueError('assets_root must be set for task, '
                             'call set_assets_root().')
        self.goals = []
        self.lang_goals = []
        self.lang_video_goals = []
        self.progress = 0  # Task progression metric in range [0, 1].
        self._rewards = 0  # Cumulative returned rewards.

    # -------------------------------------------------------------------------
    # Oracle Agent
    # -------------------------------------------------------------------------

    def oracle(self, env):
        """Oracle agent."""
        OracleAgent = collections.namedtuple('OracleAgent', ['act'])

        def act(obs, info):  # pylint: disable=unused-argument
            """Calculate action."""

            # Oracle uses perfect RGB-D orthographic images and segmentation.
            _, hmap, obj_mask = self.get_true_image(env)

            # Unpack next goal step.
            objs, matches, targs, replace, rotations, _, _, _ = self.goals[0]

            # Match objects to targets without replacement.
            if not replace:

                # Modify a copy of the match matrix.
                matches = matches.copy()

                # Ignore already matched objects.
                for i in range(len(objs)):
                    object_id, (symmetry, _) = objs[i]
                    pose = p.getBasePositionAndOrientation(object_id)
                    targets_i = np.argwhere(matches[i, :]).reshape(-1)
                    for j in targets_i:
                        if self.is_match(pose, targs[j], symmetry):
                            matches[i, :] = 0
                            matches[:, j] = 0

            # Get objects to be picked (prioritize farthest).
            nn_dists = []
            nn_targets = []
            for i in range(len(objs)):
                object_id, (symmetry, _) = objs[i]
                xyz, _ = p.getBasePositionAndOrientation(object_id)
                targets_i = np.argwhere(matches[i, :]).reshape(-1)
                if len(targets_i) > 0:
                    targets_xyz = np.float32([targs[j][0] for j in targets_i])
                    dists = np.linalg.norm(
                        targets_xyz - np.float32(xyz).reshape(1, 3), axis=1)
                    nn = np.argmin(dists)
                    nn_dists.append(dists[nn])
                    nn_targets.append(targets_i[nn])

                # Handle ignored objects.
                else:
                    nn_dists.append(0)
                    nn_targets.append(-1)
            order = np.argsort(nn_dists)[::-1]

            # Filter out matched objects.
            order = [i for i in order if nn_dists[i] > 0]

            pick_mask = None
            for pick_i in order:
                pick_mask = np.uint8(obj_mask == objs[pick_i][0])

                # Erode to avoid picking on edges.
                # pick_mask = cv2.erode(pick_mask, np.ones((3, 3), np.uint8))

                if np.sum(pick_mask) > 0:
                    break

            # Trigger task reset if no object is visible.
            if pick_mask is None or np.sum(pick_mask) == 0:
                self.goals = []
                self.lang_goals = []
                print('Object for pick is not visible. Skipping demo.')
                return

            # Get picking pose.
            pick_prob = np.float32(pick_mask)
            pick_pix = utils.sample_distribution(pick_prob)
            # For "deterministic" demonstrations on insertion-easy, use this:
            # pick_pix = (160,80)
            pick_pos = utils.pix_to_xyz(pick_pix, hmap,
                                        self.bounds, self.pix_size)
            pick_pose = (np.asarray(pick_pos), np.asarray((0, 0, 0, 1)))

            # Get placing pose.
            targ_pose = targs[nn_targets[pick_i]]
            obj_pose = p.getBasePositionAndOrientation(objs[pick_i][0])
            if not self.sixdof:
                obj_euler = utils.quatXYZW_to_eulerXYZ(obj_pose[1])
                obj_quat = utils.eulerXYZ_to_quatXYZW((0, 0, obj_euler[2]))
                obj_pose = (obj_pose[0], obj_quat)
            world_to_pick = utils.invert(pick_pose)
            obj_to_pick = utils.multiply(world_to_pick, obj_pose)
            pick_to_obj = utils.invert(obj_to_pick)
            place_pose = utils.multiply(targ_pose, pick_to_obj)

            # Rotate end effector?
            if not rotations:
                place_pose = (place_pose[0], (0, 0, 0, 1))

            place_pose = (np.asarray(place_pose[0]), np.asarray(place_pose[1]))

            return {'pose0': pick_pose, 'pose1': place_pose}

        return OracleAgent(act)

    # -------------------------------------------------------------------------
    # Reward Function and Task Completion Metrics
    # -------------------------------------------------------------------------

    def reward(
        self,
        oracle=False,
        datagen=False,
        done=False,
        obj_mask=None,
        done_multitask=False
        ):
        """
        Get delta rewards for current timestep.

        Args:
            oracle: if True, we assume access to task completion info
                i.e., an oracle tells us how we're doing and
                whether the goal is achieved

        Returns:
            A tuple consisting of the scalar (delta) reward.
        """
        # Unpack next goal step.
        objs, matches, targs, _, _, metric, params, max_reward = self.goals[0]


        # while generating data, enforce a stricter reward function
        soft_metrics = ["constraint", "multi_relation", "polygon", "line"]
        if datagen and metric in soft_metrics:
            metric = 'pose'

        # Evaluate by matching object poses.
        if metric == 'pose':
            step_reward = 0
            # loop over the objects
            for i in range(len(objs)):
                object_id, (symmetry, _) = objs[i]
                pose = p.getBasePositionAndOrientation(object_id)
                # enumerate matched targets for this object
                targets_i = np.argwhere(matches[i, :]).reshape(-1)
                for j in targets_i:  # loop over targets
                    target_pose = targs[j]
                    # if this object matches any target we're good!
                    if self.is_match(pose, target_pose, symmetry):
                        step_reward += max_reward / len(objs)
                        break

        # Evaluate by measuring object intersection with zone.
        elif metric == 'zone':
            zone_pts, total_pts = 0, 0
            obj_pts, zones = params
            for (zone_pose, zone_size) in zones:

                # Count valid points in zone.
                for obj_id in obj_pts:
                    pts = obj_pts[obj_id]
                    obj_pose = p.getBasePositionAndOrientation(obj_id)
                    world_to_zone = utils.invert(zone_pose)
                    obj_to_zone = utils.multiply(world_to_zone, obj_pose)
                    pts = np.float32(utils.apply(obj_to_zone, pts))
                    if len(zone_size) > 1:
                        valid_pts = np.logical_and.reduce([
                            pts[0, :] > -zone_size[0] / 2,
                            pts[0, :] < zone_size[0] / 2,
                            pts[1, :] > -zone_size[1] / 2,
                            pts[1, :] < zone_size[1] / 2,
                            pts[2, :] < self.zone_bounds[2, 1]
                        ])

                    # if zone_idx == matches[obj_idx].argmax():
                    zone_pts += np.sum(np.float32(valid_pts))
                    total_pts += pts.shape[1]
            step_reward = max_reward * (zone_pts / total_pts)
              
        elif metric == "multi_relation":
            step_reward = 0
            if done_multitask:
                goal_check_info = targs[0][2]
                rel_idxs = goal_check_info["rel_idxs"]
                ref_idxs = goal_check_info["ref_idxs"]
                relations = goal_check_info["relations"]
                obj_ids = goal_check_info["obj_ids"]
                for i in range(len(relations)):
                    rel_id = obj_ids[rel_idxs[i]]
                    ref_id = obj_ids[ref_idxs[i]]
                    try:
                        binary_relations = self.get_binary_relations(rel_id, ref_id, obj_mask)
                    except Exception as e:
                        print(e)
                        # this can happen if an object is placed on top of another object
                        binary_relations = []
                    if relations[i] in binary_relations:
                        step_reward += max_reward / len(relations)
        
        elif metric == "polygon":
            step_reward = 0
            if done_multitask:
                bboxs = []
                for obj_id in objs:
                    try:
                        bbox = self.get_box_from_obj_id(
                            obj_id[0], obj_mask)
                        bboxs.append(bbox)
                    except Exception as e:
                        print(e)
                bboxs = np.array(bboxs)
                bboxs = np.concatenate([
                    (bboxs[:, :2] + bboxs[:, 2:]) / 2,
                    (bboxs[:, 2:] - bboxs[:, :2])
                ], -1)
                centers = bboxs[:, :2] / np.max(obj_mask.shape)
                # Find centroid
                centroid = centers.mean(0)
                dists = np.sqrt(np.sum((centers - centroid[None]) ** 2, 1))
                dist_var = dists.std()
                if dist_var < 0.03:
                    step_reward = 1.0
                elif dist_var < 0.06:
                    step_reward = (0.06 - dist_var) / 0.03
                else:
                    step_reward = 0.0

        elif metric == "line":
            step_reward = 0
            if done_multitask:
                bboxs = []
                for obj_id in objs:
                    try:
                        bbox = self.get_box_from_obj_id(
                            obj_id[0], obj_mask)
                        bboxs.append(bbox)
                    except Exception as e:
                        print(e)
                bboxs = np.array(bboxs)
                bboxs = np.concatenate([
                    (bboxs[:, :2] + bboxs[:, 2:]) / 2,
                    (bboxs[:, 2:] - bboxs[:, :2])
                ], -1)
                centers = bboxs[:, :2] / np.max(obj_mask.shape)
                c = centers[:, 1].mean()
                
                dists = abs(centers[:, 1] - c)
                dist_var = dists.std()
                if dist_var < 0.03:
                    step_reward = 1.0
                elif dist_var < 0.06:
                    step_reward = (0.06 - dist_var) / 0.03
                else:
                    step_reward = 0.0
        
        else:
            assert False, f"{metric} not implemented"
            
        if datagen and metric == "multi_relation" and done_multitask:
            step_reward_ = 0
            goal_check_info = targs[0][2]
            rel_idxs = goal_check_info["rel_idxs"]
            ref_idxs = goal_check_info["ref_idxs"]
            relations = goal_check_info["relations"]
            obj_ids = goal_check_info["obj_ids"]

            for i in range(len(relations)):
                rel_id = obj_ids[rel_idxs[i]]
                ref_id = obj_ids[ref_idxs[i]]

                binary_relations = self.get_binary_relations(rel_id, ref_id, env)
                if relations[i] in binary_relations:
                    step_reward_ += max_reward / len(objs)

            # because constraint is actually a softer constraint than pose
            assert step_reward <= step_reward_

        # Move to next goal step
        reward = self.progress + step_reward - self._rewards  # this action
        self._rewards = self.progress + step_reward  # total
        if (not oracle or np.abs(max_reward - step_reward) < 0.01) and (done or oracle):
            if len(self.lang_goals) > 0:
                self.lang_goals.pop(0)
                self.goals.pop(0)
            self.progress += step_reward
        return reward

    def done(self):
        """
        Check if the task is done or has failed.

        Returns:
            True if the episode should be considered a success.
        """
        return (len(self.goals) == 0) or (self._rewards > 0.99)

    # -------------------------------------------------------------------------
    # Environment Helper Functions
    # -------------------------------------------------------------------------

    def is_match(self, pose0, pose1, symmetry):
        """Check if pose0 and pose1 match within a threshold."""

        # Get translational error.
        diff_pos = np.float32(pose0[0][:2]) - np.float32(pose1[0][:2])
        dist_pos = np.linalg.norm(diff_pos)

        # Get rotational error around z-axis (account for symmetries).
        diff_rot = 0
        if symmetry > 0:
            rot0 = np.array(utils.quatXYZW_to_eulerXYZ(pose0[1]))[2]
            rot1 = np.array(utils.quatXYZW_to_eulerXYZ(pose1[1]))[2]
            diff_rot = np.abs(rot0 - rot1) % symmetry
            if diff_rot > (symmetry / 2):
                diff_rot = symmetry - diff_rot

        return (dist_pos < self.pos_eps) and (diff_rot < self.rot_eps)

    def get_true_image(self, env):
        """Get RGB-D orthographic heightmaps and segmentation masks."""

        # Capture near-orthographic RGB-D images and segmentation masks.
        color, depth, segm = env.render_camera(self.oracle_cams[0])

        # Combine color with masks for faster processing.
        color = np.concatenate((color, segm[Ellipsis, None]), axis=2)

        # Reconstruct real orthographic projection from point clouds.
        hmaps, cmaps = utils.reconstruct_heightmaps(
            [color], [depth], self.oracle_cams, self.bounds, self.pix_size)

        # Split color back into color and masks.
        cmap = np.uint8(cmaps)[0, Ellipsis, :3]
        hmap = np.float32(hmaps)[0, Ellipsis]
        mask = np.int32(cmaps)[0, Ellipsis, 3:].squeeze()
        return cmap, hmap, mask

    # def get_box_from_obj_id(self, obj_id, obj_mask):
    #     obj_loc = np.where(obj_mask == obj_id)
    #     x1, x2 = np.min(obj_loc[0]), np.max(obj_loc[0])
    #     y1, y2 = np.min(obj_loc[1]), np.max(obj_loc[1])
    #     return [x1, y1, x2, y2]

    # ##### EVAL UTILS #####
    # def spatial_relations_eval(self, rel_obj_id, ref_obj_id, obj_mask, rel):
    #     # checks if box1 is rel of box2
    #     supported_relations = ["left", "right", "above", "below"]
    #     assert rel in supported_relations



    #     x1, y1, x2, y2 = box1
    #     x1_, y1_, x2_, y2_ = box2

    #     if rel == "left":
    #         return x2_ < x1
    #     elif rel == "right":
    #         return x1_ > x2
    #     elif rel == "above":
    #         return y2_ < y1
    #     elif rel == "below":
    #         return y1_ > y2
    #     else:
    #         assert False, rel

    def get_random_pose(
            self, env, obj_size,
            constraint_fn=None, ref_obj_id=None, 
            conjugate=False, ref_relations=None,
            weak=False):
        """Get random collision-free object pose within workspace bounds."""

        # Get erosion size of object in pixels.
        max_size = np.sqrt(obj_size[0] ** 2 + obj_size[1] ** 2)
        erode_size = int(np.round(max_size / self.pix_size)) + 5

        _, hmap, obj_mask = self.get_true_image(env)

        # Randomly sample an object pose within free-space pixels.
        free = np.ones(obj_mask.shape, dtype=np.uint8)
        free[0, :], free[:, 0], free[-1, :], free[:, -1] = 0, 0, 0, 0
        for obj_ids in env.obj_ids.values():
            for obj_id in obj_ids:
                free[obj_mask == obj_id] = 0

        # constraint is a binary image with value 1 where it satisfies contraint
        # and 0 otheriwse
        if constraint_fn != None:
            constraint_image = constraint_fn(
                obj_mask, ref_obj_id, conjugate=conjugate, 
                ref_relations=ref_relations
            )

            # ensure that constraint doesn't creates failure of free space
            if weak:
                free_ = free * constraint_image.astype(np.uint8)
                free_ = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))
                if np.sum(free_) == 0:
                    constraint_image = np.ones_like(free)

            free = free * constraint_image.astype(np.uint8)

        free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))
        if np.sum(free) == 0:  # no free space for this object
            return None, None

        # There is free space, place it at a random location/pose.
        pix = utils.sample_distribution(np.float32(free))
        pos = utils.pix_to_xyz(pix, hmap, self.bounds, self.pix_size)
        pos = (pos[0], pos[1], obj_size[2] / 2)
        theta = np.random.rand() * 2 * np.pi
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
        return pos, rot

    def get_lang_goal(self):
        if len(self.lang_goals) == 0:
            return self.task_completed_desc
        else:
            return self.lang_goals[0]

    def get_reward(self):
        return float(self._rewards)

    # -------------------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------------------

    def fill_template(self, template, replace):
        """Read a file and replace key strings."""
        full_template_path = os.path.join(self.assets_root, template)
        with open(full_template_path, 'r') as file:
            fdata = file.read()
        for field in replace:
            for i in range(len(replace[field])):
                fdata = fdata.replace(f'{field}{i}', str(replace[field][i]))
        # alphabet = string.ascii_lowercase + string.digits
        # rname = ''.join(random.choices(alphabet, k=16))
        rname = 'a' + str(time.time())
        tmpdir = tempfile.gettempdir()
        template_filename = os.path.split(template)[-1]
        fname = os.path.join(tmpdir, f'{template_filename}.{rname}')
        with open(fname, 'w') as file:
            file.write(fdata)
        return fname

    def get_random_size(self, min_x, max_x, min_y, max_y, min_z, max_z):
        """Get random box size."""
        size = np.random.rand(3)
        size[0] = size[0] * (max_x - min_x) + min_x
        size[1] = size[1] * (max_y - min_y) + min_y
        size[2] = size[2] * (max_z - min_z) + min_z
        return tuple(size)

    def get_box_object_points(self, obj):
        obj_shape = p.getVisualShapeData(obj)
        obj_dim = obj_shape[0][3]
        obj_dim = tuple(d for d in obj_dim)
        xv, yv, zv = np.meshgrid(
            np.arange(-obj_dim[0] / 2, obj_dim[0] / 2, 0.02),
            np.arange(-obj_dim[1] / 2, obj_dim[1] / 2, 0.02),
            np.arange(-obj_dim[2] / 2, obj_dim[2] / 2, 0.02),
            sparse=False, indexing='xy'
        )
        return np.vstack((
            xv.reshape(1, -1),
            yv.reshape(1, -1),
            zv.reshape(1, -1)
        ))

    def get_mesh_object_points(self, obj):
        mesh = p.getMeshData(obj)
        mesh_points = np.array(mesh[1])
        mesh_dim = np.vstack((mesh_points.min(0), mesh_points.max(0)))
        xv, yv, zv = np.meshgrid(
            np.arange(mesh_dim[0][0], mesh_dim[1][0], 0.02),
            np.arange(mesh_dim[0][1], mesh_dim[1][1], 0.02),
            np.arange(mesh_dim[0][2], mesh_dim[1][2], 0.02),
            sparse=False, indexing='xy'
        )
        return np.vstack((
            xv.reshape(1, -1),
            yv.reshape(1, -1),
            zv.reshape(1, -1)
        ))

    def color_random_brown(self, obj):
        shade = np.random.rand() + 0.5
        color = np.float32([shade * 156, shade * 117, shade * 95, 255]) / 255
        p.changeVisualShape(obj, -1, rgbaColor=color)

    def set_assets_root(self, assets_root):
        self.assets_root = assets_root
