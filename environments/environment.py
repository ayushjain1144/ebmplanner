"""Environment class."""

import os
import tempfile
import time

import cv2
import imageio
from PIL import Image
import textwrap

import gym
import numpy as np

from global_vars import PIXEL_SIZE, CAMERA_CONFIG, BOUNDS
from utils import pybullet_utils
import utils.transporter_utils as utils

import pybullet as p

import wandb
import ipdb
st = ipdb.set_trace

PLACE_STEP = 0.0003
PLACE_DELTA_THRESHOLD = 0.005

UR5_URDF_PATH = 'ur5/ur5.urdf'
UR5_WORKSPACE_URDF_PATH = 'ur5/workspace.urdf'
PLANE_URDF_PATH = 'plane/plane.urdf'
UR5_WORKSPACE_TMP_PATH = 'ur5/workspace_temp.urdf'


class Environment(gym.Env):
    """OpenAI Gym-style environment class."""

    def __init__(self,
                 assets_root,
                 task=None,
                 disp=False,
                 shared_memory=False,
                 hz=240,
                 record_cfg=None,
                 constant_bg=False,
                 debug=False,
                 overhead=False):
        """
        Creates OpenAI Gym-style environment with PyBullet.

        Args:
          assets_root: root directory of assets.
          task: the task to use. If None, the user must call set_task for the
            environment to work properly.
          disp: show environment with PyBullet's built-in display viewer.
          shared_memory: run with shared memory.
          hz: PyBullet physics simulation step speed.
            Set to 480 for deformables.
          constant_bg: do not randomize background for testing

        Raises:
          RuntimeError: if pybullet cannot load fileIOPlugin.
        """
        self.constant_bg = constant_bg
        self.pix_size = PIXEL_SIZE
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': [], 'move': [], 'move_goal': []}
        self.homej = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.agent_cams = CAMERA_CONFIG
        self.record_cfg = record_cfg
        self.save_video = False
        self.overhead = overhead
        self.step_counter = 0
        self.pick_map = None
        self.place_map = None
        self.debug = debug
        self.failed_datagen = False

        if self.debug:
            wandb.init(project="NS_Transporter", name="2d_vis")

        self.assets_root = assets_root
        self.name_of_obj = None

        color_tuple = [
            gym.spaces.Box(0, 255, config['image_size'] + (3,), dtype=np.uint8)
            for config in self.agent_cams
        ]
        depth_tuple = [
            gym.spaces.Box(0.0, 20.0, config['image_size'], dtype=np.float32)
            for config in self.agent_cams
        ]
        self.observation_space = gym.spaces.Dict({
            'color': gym.spaces.Tuple(color_tuple),
            'depth': gym.spaces.Tuple(depth_tuple),
        })
        self.position_bounds = gym.spaces.Box(
            low=BOUNDS[:, 0],
            high=BOUNDS[:, 1],
            shape=(3,),
            dtype=np.float32)
        self.action_space = gym.spaces.Dict({
            'pose0':
                gym.spaces.Tuple(
                    (self.position_bounds,
                     gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32))),
            'pose1':
                gym.spaces.Tuple(
                    (self.position_bounds,
                     gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)))
        })

        # Start PyBullet.
        disp_option = p.DIRECT
        if disp:
            disp_option = p.GUI
            if shared_memory:
                disp_option = p.SHARED_MEMORY
        client = p.connect(disp_option)
        file_io = p.loadPlugin('fileIOPlugin', physicsClientId=client)
        if file_io < 0:
            raise RuntimeError('pybullet: cannot load FileIO!')
        if file_io >= 0:
            p.executePluginCommand(
                file_io,
                textArgument=assets_root,
                intArgs=[p.AddFileIOAction],
                physicsClientId=client)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.setAdditionalSearchPath(assets_root)
        p.setAdditionalSearchPath(tempfile.gettempdir())
        p.setTimeStep(1. / hz)

        self.setseed = 0
        # If using --disp, move default camera closer to the scene.
        if disp:
            target = p.getDebugVisualizerCamera()[11]
            p.resetDebugVisualizerCamera(
                cameraDistance=1.1,
                cameraYaw=90,
                cameraPitch=-25,
                cameraTargetPosition=target)

        if task:
            self.set_task(task)

    def __del__(self):
        if hasattr(self, 'video_writer'):
            self.video_writer.close()

    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        v = [np.linalg.norm(p.getBaseVelocity(i)[0])
             for i in self.obj_ids['rigid']]
        return all(np.array(v) < 5e-3)

    def add_object(self, urdf, pose, category='rigid', dummy=False):
        """List of (fixed, rigid, or deformable) objects in env."""
        fixed_base = 1 if category == 'fixed' else 0
        obj_id = pybullet_utils.load_urdf(
            p,
            os.path.join(self.assets_root, urdf),
            pose[0],
            pose[1],
            useFixedBase=fixed_base)
        if obj_id is not None and not dummy:
            self.obj_ids[category].append(obj_id)
        return obj_id

    # ---------------------------------------------------------------------------
    # Standard Gym Functions
    # ---------------------------------------------------------------------------

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        self.setseed = seed
        return seed

    def fill_template(self, template, replace):
        """Read a file and replace key strings."""
        full_template_path = os.path.join(self.assets_root, template)
        with open(full_template_path, 'r') as file:
            fdata = file.read()
        for field in replace:
            for i in range(len(replace[field])):
                fdata = fdata.replace(f'{field}{i}', str(replace[field][i]))
        tmpdir = 'tmp'  # tempfile.gettempdir()
        template_filename = os.path.split(template)[-1]
        fname = os.path.join(tmpdir, f'{template_filename}')
        with open(fname, 'w') as file:
            file.write(fdata)
        return fname

    def reset(self):
        """Performs common reset functionality for all supported tasks."""
        if not self.task:
            raise ValueError('environment task must be set')
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': [], 'move': [], 'move_goal': []}
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setGravity(0, 0, -9.8)

        # Temporarily disable rendering to load scene faster.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        pybullet_utils.load_urdf(
            p, os.path.join(self.assets_root, PLANE_URDF_PATH),
            [0, 0, -0.001]
        )

        if self.task.mode != 'test' or self.constant_bg:
            pybullet_utils.load_urdf(
                p, os.path.join(self.assets_root, UR5_WORKSPACE_URDF_PATH),
                [0.5, 0, 0]
            )
        else:  # randomize background in a pseudo-random manner
            col = (1 + np.arange(3)) * (self.setseed + np.arange(3)) % 12 / 11
            urdf = self.fill_template(
                UR5_WORKSPACE_TMP_PATH,
                {'COL': (col[0], col[1], col[2], 1)}
            )
            pybullet_utils.load_urdf(p, urdf, [0.5, 0, 0])
        # Load UR5 robot arm equipped with suction end effector.
        # TODO(andyzeng): add back parallel-jaw grippers.
        self.ur5 = pybullet_utils.load_urdf(
            p, os.path.join(self.assets_root, UR5_URDF_PATH))
        self.ee = self.task.ee(self.assets_root, self.ur5, 9, self.obj_ids)
        self.ee_tip = 10  # Link ID of suction cup.

        # Get revolute joint indices of robot (skip fixed joints).
        n_joints = p.getNumJoints(self.ur5)
        joints = [p.getJointInfo(self.ur5, i) for i in range(n_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

        # Move robot to home joint configuration.
        for i in range(len(self.joints)):
            p.resetJointState(self.ur5, self.joints[i], self.homej[i])

        targj = np.array([
            -3.5653608, -1.1199822, 0.02722096,
            -0.50724125, -1.5674078, -0.4235661
        ])
        self.movej(targj, 0.01)
        # Reset end effector.
        self.ee.release()

        # Reset task.
        self.name_of_obj = self.task.reset(self)

        if self.failed_datagen:
            print("Failed datagen")
            return None

        # Re-enable rendering.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        obs, _, _, _ = self.step()
        return obs

    def step(self, action=None, oracle=False):
        """
        Execute action with specified primitive.

        Args:
          action: action to execute.

        Returns:
          (obs, reward, done, info) tuple containing MDP step data.
        """
        # self.pick_map = pick_map
        # self.place_map = place_map
        if action is not None:
            timeout = self.task.primitive(
                self.movej, self.movep, self.ee,
                action['pose0'], action['pose1']
            )

            # Exit early if action times out. We still return an observation
            # so that we don't break the Gym API contract.
            if timeout:
                obs = {'color': (), 'depth': ()}
                for config in self.agent_cams:
                    color, depth, _ = self.render_camera(config)
                    obs['color'] += (color,)
                    obs['depth'] += (depth,)
                _, hmap = utils.get_fused_heightmap(
                    obs, self.agent_cams, self.task.bounds,
                    pix_size=self.pix_size
                )
                obs['height'] = hmap
                return obs, 0.0, True, self.info

        # Step simulator asynchronously until objects settle.
        while not self.is_static:
            self.step_simulation()

        if oracle:
            reward = self.task.reward(oracle=oracle) if action is not None else 0
            info = {}
            info.update(self.info)
        else:
            reward = 0.0

        # Get task rewards.
        done = self.task.done()

        obs = self._get_obs()

        return obs, reward, done, self.info

    def step_simulation(self):
        p.stepSimulation()
        self.step_counter += 1

        if (self.save_video or self.debug) and self.step_counter % 5 == 0:
            self.add_video_frame()

    def render(self, mode='rgb_array'):
        # Render only the color image from the first camera.
        # Only support rgb_array for now.
        if mode != 'rgb_array':
            raise NotImplementedError('Only rgb_array implemented')
        color, _, _ = self.render_camera(self.agent_cams[0])
        return color

    def render_camera(self, config, image_size=None, shadow=0):
        """Render RGB-D image with specified camera configuration."""
        if not image_size:
            image_size = config['image_size']

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config['rotation'])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config['position'] + lookdir
        focal_len = config['intrinsics'][0]
        znear, zfar = config['zrange']
        viewm = p.computeViewMatrix(config['position'], lookat, updir)
        fovh = (image_size[0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = image_size[1] / image_size[0]
        projm = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = p.getCameraImage(
            width=image_size[1],
            height=image_size[0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=shadow,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=p.ER_BULLET_HARDWARE_OPENGL)
        # Get color image.
        color_image_size = (image_size[0], image_size[1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if config['noise']:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, image_size))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (image_size[0], image_size[1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
        depth = (2. * znear * zfar) / depth
        if config['noise']:
            depth += self._random.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)

        return color, depth, segm

    def xyz_to_pix2(self, position):
        """Convert from 3D position to pixel location on heightmap."""
        u = int(np.round((position[1] - BOUNDS[1, 0]) / self.pix_size))
        v = int(np.round((position[0] - BOUNDS[0, 0]) / self.pix_size))
        return (u, v)

    def xyz_to_pix(self, minpoint, maxpoint):
        """Convert from 3D position to pixel location on heightmap."""
        left = self.xyz_to_pix2(minpoint)
        right = self.xyz_to_pix2(maxpoint)
        return left, right

    def update_info_online(self, info):
        for k, _ in info.items():
            if k == 'names' or k == 'lang_goal':
                continue
            for key, value in info[k].items():
                left, right = self.xyz_to_pix(value[2][0], value[2][1])
                rot = value[1]
                if info['names'] is None:
                    info[k][key] = ((left, right), rot)
                else:
                    info[k][key] = ((left, right), rot, info['names'][key])

        return info

    @property
    def info(self):
        """Environment info with object poses, dimensions, and colors."""
        info = {}  # object id : (position, rotation, dimensions)
        if self.task.goals == []:
            self.obj_ids['move'] = []
        elif self.task.goals is not None: #and len(self.task.goals[0][0]) == 1:
            self.obj_ids['move'] = [self.task.goals[0][0][i][0] for i in range(len(self.task.goals[0][0]))]
            self.obj_ids['move_goal'] = [self.task.goals[0][0][i][0] for i in range(len(self.task.goals[0][0]))]
        for (obj_k, obj_ids) in self.obj_ids.items():
            # import ipdb;ipdb.set_trace()
            info[obj_k] = {}
            for obj_id in obj_ids:
                pos, rot = p.getBasePositionAndOrientation(obj_id)
                # dim = p.getVisualShapeData(obj_id)[0][3]
                dim = p.getAABB(obj_id)
                info[obj_k][obj_id] = (pos, rot, dim)

        if 'assembling-kits' in self.task.name and len(self.task.goals) > 0:
            info['deformable'] = info['fixed']
            for k, v in info['fixed'].items():
                a = v[0]
                b = self.task.goals[0][2][0][0]
                if sum(abs(np.array(a) - np.array(b))) < 1e-3:
                    info['fixed'] = {}
                    info['fixed'][k] = v
                    break
        info['lang_goal'] = self.get_lang_goal()
        info['names'] = self.name_of_obj
        return self.update_info_online(info)

    def set_task(self, task):
        task.set_assets_root(self.assets_root)
        self.task = task

    def set_failed_dategen(self, failed_datagen):
        self.failed_datagen = failed_datagen

    def get_lang_goal(self):
        if self.task:
            return self.task.get_lang_goal()
        else:
            raise Exception("No task for was set")

    # -----------------------------------------------------------------------
    # Robot Movement Functions
    # -----------------------------------------------------------------------

    def movej(self, targj, speed=0.01, timeout=5):
        """Move UR5 to target joint configuration."""
        if self.save_video:
            timeout = timeout * 50

        t0 = time.time()
        while (time.time() - t0) < timeout:
            currj = [p.getJointState(self.ur5, i)[0] for i in self.joints]
            currj = np.array(currj)
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return False

            # Move with constant velocity
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints))
            p.setJointMotorControlArray(
                bodyIndex=self.ur5,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains)
            self.step_counter += 1
            self.step_simulation()

        print(f'Warning: movej exceeded {timeout} second timeout. Skipping.')
        return True

    def start_rec(self, video_filename):
        assert self.record_cfg

        # video_filename += ' .'.join(self.task.lang_goals)

        # make video directory
        if not os.path.exists(self.record_cfg['save_video_path']):
            os.makedirs(self.record_cfg['save_video_path'])

        # close and save existing writer
        if hasattr(self, 'video_writer'):
            self.video_writer.close()

        # initialize writer
        self.video_writer = imageio.get_writer(
            os.path.join(
                self.record_cfg['save_video_path'],
                f"{video_filename}.mp4"
            ),
            fps=self.record_cfg['fps'],
            format='FFMPEG',
            codec='h264'
        )
        p.setRealTimeSimulation(False)
        self.save_video = True

    def end_rec(self):
        if hasattr(self, 'video_writer'):
            self.video_writer.close()

        p.setRealTimeSimulation(True)
        self.save_video = False


    def add_video_frame_text(self, caption=None, H=320, W=640):
        color = np.zeros((H, W, 3)).astype(np.float32)
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.65
        font_thickness = 1

        image_size = color.shape
        wrapped_text = textwrap.wrap(caption, width=40)
        for i, line in enumerate(wrapped_text):
            textsize = cv2.getTextSize(
                    line, font, font_scale, font_thickness
                )[0]
            gap = textsize[1] + 9

            textX = (image_size[1] - textsize[0]) // 2
            textY = int((image_size[0] + textsize[1]) / 2) + i * gap - (len(wrapped_text) // 2) * gap

            color = color.copy()
            color = cv2.putText(color, line, org=(textX, textY),
                                fontScale=font_scale,
                                fontFace=font,
                                color=(255, 255, 255),
                                thickness=font_thickness, lineType=cv2.LINE_AA)
        color = np.array(color)
        color = np.uint8(np.round(color))

        for i in range(self.record_cfg['fps'] * 3):
            self.video_writer.append_data(color)


    def add_video_frame_executor(self, color=None, caption=None, boxes=None, num_repeats=25, font_scale=0.65):
        """
        color: H X W X 3
        """
        # image_size = (
        #     self.record_cfg['video_height'], self.record_cfg['video_width']
        # )
        # config = self.agent_cams[3]
        # color, _, _ = self.render_camera(config, image_size, shadow=0)
        if color is None:
            obs = self._get_obs()
            color, _ = utils.get_fused_heightmap(
                obs, self.agent_cams, self.task.bounds, pix_size=self.pix_size
            )
            color = color.transpose(1, 0, 2)

        color = np.array(color[..., :3]).astype(np.float32)

        font = cv2.FONT_HERSHEY_DUPLEX
        font_thickness = 1
        if boxes is not None:
            for box in boxes:
                color = color.copy()
                color = cv2.rectangle(color, (box[0], box[1]),
                    (box[2], box[3]),
                    (255, 0, 0),
                    thickness=2)
        if caption is not None:
            image_size = color.shape
            textsize = cv2.getTextSize(
                    caption, font, font_scale, font_thickness
                )[0]
            textX = (image_size[1] - textsize[0]) // 2
            color = color.copy()
            color = cv2.putText(color, caption, org=(textX, 300),
                                fontScale=font_scale,
                                fontFace=font,
                                color=(255, 255, 255),
                                thickness=font_thickness, lineType=cv2.LINE_AA)
        color = np.array(color)
        color = np.uint8(np.round(color))

        for i in range(num_repeats):
            self.video_writer.append_data(color)


    def add_video_frame(self):
        # Render frame.
        image_size = (
            self.record_cfg['video_height'], self.record_cfg['video_width']
        )
        if not self.overhead:
            config = self.agent_cams[0]
            color_, _, _ = self.render_camera(
                config,
                image_size,
                shadow=0
            )
            color_ = color_
            color_ = cv2.resize(
                color_.astype(np.float32),
                (image_size[1] // 3, image_size[0] // 3),
                interpolation = cv2.INTER_AREA)
            color_ = np.array(color_)
            

            config = self.agent_cams[3]
            color, _, _ = self.render_camera(config, (640, 720), shadow=0)
            color = color
            color = cv2.resize(
                color.astype(np.float32),
                (image_size[1], image_size[0]),
                interpolation = cv2.INTER_AREA)
            color = np.array(color)

            color[:image_size[0] // 3, (-image_size[1] // 3 + 1):] = color_
        else:
            obs = self._get_obs()
            color, _ = utils.get_fused_heightmap(
                obs, self.agent_cams, self.task.bounds, pix_size=self.pix_size
            )
            color = color.transpose(1, 0, 2)
        color = np.array(color[..., :3]).astype(np.float32)

        # Add language instruction to video.
        if self.record_cfg['add_text']:
            lang_goal = self.get_lang_goal()
            reward = f"Success: {self.task.get_reward():.3f}"

            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.65
            font_thickness = 1

            # Write language goal.
            lang_textsize = cv2.getTextSize(
                lang_goal, font, font_scale, font_thickness
            )[0]
            lang_textX = (image_size[1] - lang_textsize[0]) // 2
            color = color.copy()
            color = cv2.putText(color, lang_goal, org=(lang_textX, 600),
                                fontScale=font_scale,
                                fontFace=font,
                                color=(0, 0, 0),
                                thickness=font_thickness, lineType=cv2.LINE_AA)

            # Write Reward.
            reward_textsize = cv2.getTextSize(
                reward, font, font_scale, font_thickness
            )[0]
            reward_textX = (image_size[1] - reward_textsize[0]) // 2
            color = cv2.putText(color, reward, org=(reward_textX, 634),
                                fontScale=font_scale,
                                fontFace=font,
                                color=(0, 0, 0),
                                thickness=font_thickness, lineType=cv2.LINE_AA)
            if self.debug:
                wandb.log({"image": wandb.Image(Image.fromarray(color))})
                return None
                
        color = np.array(color)
        color = np.uint8(np.round(color))
        self.video_writer.append_data(color)

    def movep(self, pose, speed=0.01):
        """Move UR5 to target end effector pose."""
        targj = self.solve_ik(pose)
        return self.movej(targj, speed)

    def solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.ur5,
            endEffectorLinkIndex=self.ee_tip,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
            upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
            restPoses=np.float32(self.homej).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5)
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

    def _get_obs(self):
        # Get RGB-D camera image observations.
        obs = {'color': (), 'depth': ()}
        for config in self.agent_cams:
            color, depth, _ = self.render_camera(config)
            obs['color'] += (color,)
            obs['depth'] += (depth,)
        _, hmap = utils.get_fused_heightmap(
            obs, self.agent_cams, self.task.bounds, pix_size=self.pix_size
        )
        obs['height'] = hmap

        return obs
