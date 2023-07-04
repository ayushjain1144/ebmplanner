"""Pipeline for parser training/testing."""

import argparse
import io
import os
import os.path as osp
import random
import math

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from beauty_detr.cfg import get_bdetr_args
from data.transporter_dataset import get_image, RavensDataset
from environments.environment import Environment
from init_executor import _initialize_executor
from models.transporter_torch import (
    OriginalTransporterAgent, \
    GCTransporterAgent
)
import tasks
from utils.executor_utils import _load_gt_boxes, flip_data, flip_heatmaps, mask_image_with_boxes, flip_boxes
from data.create_cliport_programs import utterance2program_bdetr

import wandb
import ipdb
st = ipdb.set_trace


class Trainer:
    """Train/test models on manipulation."""

    def __init__(self, model, data_loaders, args):
        self.model = model
        self.data_loaders = data_loaders
        self.args = args

        self.writer = SummaryWriter(f'runs/{args.tensorboard_dir}')
        self.pk_optimizer = Adam(model.pick_net.parameters(), lr=args.lr)
        self.pl_optimizer = Adam(model.place_net.parameters(), lr=args.lr)
        self.env = _set_environment(args) if args.eval_on_env else None
        self.executor = None
        if args.eval_with_executor:
            self.executor = _initialize_executor(args)
        wandb.init(project="NS_Transporter", name="iclr_")

        self.move_all_task = ['composition-seen-colors-group', 'circle-seen-colors', 'line-seen-colors']
        unseen_tasks = [t.replace("seen", "unseen") for t in self.move_all_task]
        self.move_all_task.extend(unseen_tasks)

    def run(self):
        # Set
        start_epoch = 0
        val_acc_prev_best = -1.0

        # Load
        if osp.exists(self.args.ckpnt):
            start_epoch, val_acc_prev_best = self._load_ckpnt()
            val_acc_prev_best = -1.0
            print("Loaded checkpoint")

        # Eval?
        if self.args.eval:
            self.model.eval()
            # self.train_test_loop('test')
            if self.args.eval_all:
                test_acc = 0.0
                for i, task_ in enumerate(self.data_loaders['test'].dataset.task_list):
                    test_acc += self.evaluate_task(mode='test', eval_task=task_)
                test_acc /= (i + 1)
            else:
                test_acc = self.evaluate_task(mode='test')
            print(f"Test Accuracy: {test_acc}")
            return self.model

        # Go!
        for epoch in range(start_epoch, self.args.epochs):
            print("Epoch: %d/%d" % (epoch + 1, self.args.epochs))
            self.model.train()
            # Train
            self.train_test_loop('train', epoch)
            # Validate
            print("\nValidation")
            with torch.no_grad():
                self.train_test_loop('val', epoch)
            val_acc = 0
            if not (epoch % self.args.eval_freq):
                if self.args.eval_all:
                    for i, task_ in enumerate(self.data_loaders['val'].dataset.task_list):
                        val_acc += self.evaluate_task(epoch, mode='val', eval_task=task_)
                    val_acc /= (i + 1)
                    wandb.log({
                        "Mean Reward across all tasks": val_acc,
                        "epoch": epoch
                    })
                    print(f"Mean Validation Accuracy: {val_acc}")
                else:
                    val_acc = self.evaluate_task(epoch, mode='val')

            # Store
            if val_acc >= val_acc_prev_best or self.args.debug:
                print("Saving Checkpoint")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "pk_optimizer_state_dict": self.pk_optimizer.state_dict(),
                    "pl_optimizer_state_dict": self.pl_optimizer.state_dict(),
                    "best_acc": val_acc
                }, self.args.ckpnt)
                val_acc_prev_best = val_acc
            else:
                print("Updating Checkpoint")
                checkpoint = torch.load(self.args.ckpnt)
                checkpoint["epoch"] += 1
                torch.save(checkpoint, self.args.ckpnt)

        if self.args.eval_all:
            test_acc = 0.0
            for i, task_ in enumerate(self.data_loaders['test'].dataset.task_list):
                test_acc += self.evaluate_task(mode='test', eval_task=task_)
            test_acc /= (i + 1)
        else:
            test_acc = self.evaluate_task(mode='test')
        print(f"Test Accuracy: {test_acc}")
        return self.model

    def _load_ckpnt(self):
        ckpnt = torch.load(self.args.ckpnt)
        self.model.load_state_dict(ckpnt["model_state_dict"], strict=False)
        self.pk_optimizer.load_state_dict(ckpnt["pk_optimizer_state_dict"])
        self.pl_optimizer.load_state_dict(ckpnt["pl_optimizer_state_dict"])
        start_epoch = ckpnt["epoch"]
        val_acc_prev_best = ckpnt['best_acc']
        return start_epoch, val_acc_prev_best

    def train_test_loop(self, mode='train', epoch=1000):
        n_examples = 0
        for step, ex in tqdm(enumerate(self.data_loaders[mode])):
            image = ex['image'].to(self.args.device).float()
            if self.args.goal_conditioned_training:
                img_pick = [mask_image_with_boxes(
                    image[k],
                    ex['gt_pick_box'][k][None].to(image.device)
                ) for k in range(len(image))]
                img_pick = torch.stack(img_pick, 0)

                img_place = [mask_image_with_boxes(
                    image[k],
                    ex['gt_place_box'][k][None].to(image.device)
                ) for k in range(len(image))]
                img_place = torch.stack(img_place, 0)
            else:
                img_pick = image
                img_place = image

            p0_x = ex['p0'][:, 1].long()
            p0_y = ex['p0'][:, 0].long()
            p0_theta = ex['p0_theta']
            p0_theta = p0_theta / (2 * np.pi)
            p0_theta = torch.round(p0_theta).long()
            p1_x = ex['p1'][:, 1].long()
            p1_y = ex['p1'][:, 0].long()
            p1_theta = ex['p1_theta']
            p1_theta = p1_theta / (2 * np.pi / self.args.n_rotations)
            p1_theta = torch.round(p1_theta).long() % self.args.n_rotations
            lang = ex['lang_goal']

            # Run pick module
            pk_loss, pick, pk_labels = self._pick(
                img_pick, lang, p0_x, p0_y, p0_theta, mode
            )

            # Run place module
            pl_loss, place, pl_labels = self._place(
                img_place, lang, p0_x, p0_y, p1_x, p1_y, p1_theta,
                mode, img_pick if self.args.goal_conditioned_training else None
            )
            n_examples += len(image)

            # Logging
            try:
                wandb.log({
                    f"{mode}_pick_loss": pk_loss,
                    "epoch": epoch * len(self.data_loaders[mode]) + step
                })
                wandb.log({
                    f"{mode}_place_loss": pl_loss,
                    "epoch": epoch * len(self.data_loaders[mode]) + step
                })
            except:
                pass

            # Visualizations
            # if self.args.visualize:
            _img = _visualize(
                image[0][..., :3].cpu().numpy(),
                image[0][..., 3].cpu().numpy(),
                map_list=[
                    {
                        'pick': pk_labels.sum(-1).numpy(),
                        'place': pl_labels.sum(-1).numpy(),
                        'type': 'point', 'name': 'ground-truth'
                    },
                    {
                        'pick': pick.sum(-1).numpy(),
                        'place': place.sum(-1).numpy(),
                        'name': 'no-softmax'
                    },
                    {
                        'pick': pick.reshape(-1).softmax(0).reshape(
                            pick.shape
                        ).sum(-1).numpy(),
                        'place': place.reshape(-1).softmax(0).reshape(
                            place.shape
                        ).sum(-1).numpy(),
                        'name': 'T=1-softmax'
                    }
                ],
                text=ex['lang_goal'][0]
            )
            try:
                wandb.log({
                    f"{mode}_maps": wandb.Image(_img.permute(1, 2, 0).numpy()),
                    "epoch": epoch * len(self.data_loaders[mode]) + step
                })
            except:
                pass


    def _pick(self, image, lang, p0_x, p0_y, p0_theta, mode='train'):
        pick = self.model.pick_forward(image, lang)  # [B H W nrot]
        pk_labels = torch.zeros_like(pick, device=pick.device)
        pk_labels[range(len(p0_x)), p0_y, p0_x, p0_theta] = 1
        pk_loss = F.kl_div(
            pick.reshape(len(pick), -1).log_softmax(1),
            pk_labels.reshape(len(pick), -1),
            reduction='batchmean'
        )
        if mode == 'train':
            self.pk_optimizer.zero_grad()
            pk_loss.backward()
            self.pk_optimizer.step()
        pk_loss = pk_loss.detach().item()
        return (
            pk_loss, pick[0].detach().cpu(), pk_labels[0].detach().cpu()
        )

    def _place(self, image, lang, p0_x, p0_y, p1_x, p1_y, p1_theta,
               mode='train', img_pick=None):
        place = self.model.place_forward(image, (p0_y, p0_x), lang, img_pick=img_pick)
        pl_labels = torch.zeros_like(place, device=place.device)
        pl_labels[range(len(p1_x)), p1_y, p1_x, p1_theta] = 1
        pl_loss = F.kl_div(
            place.reshape(len(place), -1).log_softmax(1),
            pl_labels.reshape(len(place), -1),
            reduction='batchmean'
        )
        if mode == 'train':
            self.pl_optimizer.zero_grad()
            pl_loss.backward()
            self.pl_optimizer.step()
        pl_loss = pl_loss.detach().item()
        return (
            pl_loss, place[0].detach().cpu(), pl_labels[0].detach().cpu()
        )

    def evaluate_task(self, epoch=1000, mode='test', eval_task=None):
        """
        Evaluate model online.

        The environment (env) is gym-like and is expected to:
            - Called: obs, reward, done, info = env.step(act), where:
                * obs = {  # from 3 cameras
                    'color': (front (480, 640, 3), left, right),
                    'depth': (front (480, 640), left, right),
                    'height': (640, 320)
                }
                * reward (float): in [0, 1]
                * done (bool): whether task is complete
                * info = {
                    'fixed': {id: (((y1, x1), (y2, x2)), (a, b, c, d), name)}
                        where (a, b, c, d) is a quaternion
                        fixed are the objects involved in the task
                        but should not be moved
                    'rigid': same format as fixed
                        all objects - fixed
                    'deformable': same format as fixed
                        no idea what this is
                    'move': same format as fixed
                        objects involved in task and should be moved
                    'lang_goal': (str) task description
                    'names': {id: str (name) for all objects}
                }
                * act = {
                    'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
                    'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
                    'pick': p0_pix,
                    'place': p1_pix
                }
            - Reset: obs = env.reset()
            - Get/set: info = env.info, env.set_task(task)
        """
        self.model.eval()
        dset = self.data_loaders[mode].dataset
        rewards = []
        demos_with_low_reward = []

        eval_task = eval_task if eval_task is not None else self.args.eval_task
        if self.args.eval_list is not None:
            eval_list = self.args.eval_list
        else:
            eval_list = range(len(dset.seeds_per_task[eval_task]))
        for i in tqdm(eval_list):
            # Set env with seed
            name, seed = dset.get_seed_by_task_and_idx(eval_task, i)
            np.random.seed(seed)
            task = tasks.names[name]()
            task.mode = mode
            task.name = name
            self.env.seed(seed)
            random.seed(seed)
            self.env.set_task(task)
            obs = self.env.reset()
            info = self.env.info
            total_reward, step = 0, 0

            visualize_outputs = []
            columns = []
            if self.args.record:
                self.env.start_rec(f'{i+1:06d}')
            # Max steps and symbolic part
            for j in range(len(task.lang_goals)):
                step = 0
                num_steps = len(task.goals[0][0])
                max_steps = num_steps if not self.args.oracle else task.max_steps
                pick_hmap, place_hmap = None, None

                if self.args.eval_with_executor:
                    move_all = eval_task in self.move_all_task
                    if move_all:
                        place_boxes = None
                    else:
                        place_boxes = torch.tensor(
                                dset.retrieve_by_task_and_name(eval_task, f'{i:06d}-{seed}.pkl', obs_act_id=j)[-1].copy()
                            )
                    is_cliport = not (self.args.relations or self.args.multi_relations or self.args.multi_relations_group or self.args.shapes)
                    outputs = _run_executor(
                        obs, info, self.executor,
                        gt_place_boxes=place_boxes,
                        legacy=self.args.legacy,
                        is_cliport=is_cliport, 
                        move_all=move_all
                    )
                    pick_hmap, place_hmap, visualize_outputs_, columns_ = outputs['pick_hmap'], \
                        outputs['place_hmap'], outputs['visualize_outputs'], outputs['columns']
                    visualize_outputs.extend(visualize_outputs_)
                    columns.extend(columns_)
                    max_steps = len(pick_hmap)

                    if self.args.record:
                        color = get_image(obs)[..., :3]
                        color = color.transpose(1, 0, 2)

                        # write language
                        if len(task.lang_video_goals) != 0:
                            caption = task.lang_video_goals[0]
                        else:
                            caption = info['lang_goal']
                        self.env.add_video_frame_text(caption=caption)

                        # write grounding 
                        for caption, boxes in zip(outputs['video_dict']['captions'], outputs['video_dict']['boxes']):
                            self.env.add_video_frame_executor(color, f'VLMGround(image, "{caption}")', boxes) 
                        
                        # write ebm outputs
                        caption = outputs['video_dict']['ebm_captions'][0]
                        # caption = None
                        for boxes in outputs['video_dict']['ebm_boxes'][0]:
                            self.env.add_video_frame_executor(color, caption, boxes, num_repeats=2, font_scale=0.35)

                # getting the boxes from original info instead of from the inner loop
                # because info can change
                if self.args.goal_conditioned_training:
                    pick_boxes = outputs['pick_boxes']
                    place_boxes =  outputs['place_boxes']
                    
                # Execute
                reward = 0.0
                while step < max_steps:
                    done_multitask = (step == max_steps - 1)
                    print(f"i: {i}, step: {step}, lang: {info['lang_goal']}, max_steps: {max_steps}")

                    # Act
                    img = get_image(obs)
                    img = torch.from_numpy(img).to(self.args.device)

                    if self.args.goal_conditioned_training:
                        if max_steps != len(pick_boxes):
                            assert False
                        img_pick = mask_image_with_boxes(
                            img,
                            pick_boxes[step][None].to(self.args.device))
                        img_place = mask_image_with_boxes(
                            img,
                            place_boxes[step][None].to(self.args.device))
                        if self.args.visualize:
                            wandb.log({
                                "place image": wandb.Image(img_place[..., :3].cpu().numpy(), caption=info['lang_goal'])
                            })
                            wandb.log({
                                "pick image": wandb.Image(img_pick[..., :3].cpu().numpy(), caption=info['lang_goal'])
                            })
                        
                    pkmap = None if pick_hmap is None else pick_hmap[step]
                    plmap = None if place_hmap is None else place_hmap[step]
                    with torch.no_grad():
                        if self.args.goal_conditioned_training:
                            act, pick, place, tr_pick, tr_place = self.model(
                                img_pick=img_pick,
                                img_place=img_place,
                            )
                        else:
                            act, pick, place, tr_pick, tr_place = self.model(
                                img,
                                lang=[info['lang_goal']],
                                pick_hmap=pkmap,
                                place_hmap=plmap
                            )
                    pick_argmax = torch.zeros_like(pick).reshape(-1)
                    pick_argmax[pick.argmax()] = 1.0
                    pick_argmax = pick_argmax.reshape(pick.shape)
                    place_argmax = torch.zeros_like(place).reshape(-1)
                    place_argmax[place.argmax()] = 1.0
                    place_argmax = place_argmax.reshape(place.shape)
                    obs, reward_, done, info = self.env.step(act, oracle=self.args.oracle)
                    
                    if self.args.oracle:
                        print(f"Step Reward: {reward_}")
                        reward += reward_
                        print(f"Running Reward: {reward}")
                    if self.args.oracle and done:
                        print("Done!")
                        break

                    # Visualize before
                    if self.args.visualize:
                        map_list = [
                            {
                                'pick': pick_argmax.sum(-1).numpy(),
                                'place': place_argmax.sum(-1).numpy(),
                                'type': 'point', 'name': 'argmax'
                            },
                            {
                                'pick': tr_pick.sum(-1),
                                'place': tr_place.sum(-1),
                                'name': 'transporter'
                            }
                        ]
                        if self.args.eval_with_executor:
                            map_list.append({
                                'pick': pkmap.numpy(),
                                'place': plmap.numpy(),
                                'name': 'symbolic'
                            })
                            map_list.append({
                                'pick': pick.sum(-1).numpy(),
                                'place': place.sum(-1).numpy(),
                                'name': 'combined'
                            })
                        _img = _visualize(
                            img[..., :3].cpu().numpy(),
                            img[..., 3].cpu().numpy(),
                            map_list,
                            text=info['lang_goal']
                        )
                        visualize_outputs.append(
                            wandb.Image(
                                Image.fromarray(_img.permute(1, 2, 0).numpy()),
                                caption=f'before_{step}'
                            )
                        )
                        columns.append(f"before_{step}")

                        # Visualize after
                        _img = get_image(obs)
                        _img = _visualize(_img[..., :3], _img[..., 3])
                        visualize_outputs.append(
                            wandb.Image(
                                Image.fromarray(_img.permute(1, 2, 0).numpy()),
                                caption=f'after_{step}'
                            )
                        )
                        columns.append(f"after_{step}")
                    if not self.args.oracle:
                        _, _, obj_mask = task.get_true_image(self.env)
                        reward += self.env.task.reward(
                            done=False, obj_mask=obj_mask,
                            done_multitask=done_multitask
                        )
                    
                    # Update reward and step
                    step += 1
                    
                    print(f"Reward actual: {reward}")

                # just for popping out the goal and lang
                if not self.args.oracle:
                    self.env.task.reward(done=True)

                info = self.env.info
                total_reward += reward
            rewards.append(total_reward)
            if self.args.visualize:
                preview_dt = wandb.Table(columns)
                preview_dt.add_data(*visualize_outputs)
                wandb.log({f"table_{epoch}_{np.random.rand()}": preview_dt})
            assert total_reward <= 1.0
            if total_reward < 1:
                demos_with_low_reward.append(i)
            if self.args.record:
                self.env.end_rec()
        print('demos to check', demos_with_low_reward)
        mean_reward = np.mean(rewards)
        
        print(f'Mean reward {eval_task}:', mean_reward)
        wandb.log({
            f"reward {eval_task}": mean_reward,
            "epoch": epoch
        })
        # self.writer.add_scalar('reward', mean_reward, epoch)
        return mean_reward


def _set_environment(args):
    with open("cfg/eval.yaml", "r") as f:
        vcfg = yaml.load(f, Loader=yaml.FullLoader)  # val config
    env = Environment(
        vcfg['assets_root'],
        disp=vcfg['disp'],
        shared_memory=vcfg['shared_memory'],
        hz=480,
        record_cfg=vcfg['record'],
        constant_bg=args.constant_bg,
        overhead=args.overhead
    )
    return env


def _run_executor(
    obs, info, executor, gt_place_boxes=None, legacy=False, is_cliport=False, move_all=False):
    img = get_image(obs)[..., :3]
    lang = info['lang_goal']
    H, W = img.shape[:2]

    if move_all:   
        ground_truths = [
            *_load_gt_boxes(info, ['move'], H, W),
        ]
    else:
        ground_truths = [
            _load_gt_boxes(info, ['move'], H, W),
            _load_gt_boxes(info, ['fixed'], H, W),
        ]
   
    utterance2program = utterance2program_bdetr

    # flip the image and ground truths from 640X320 to 320X640
    img, ground_truths = flip_data(img, ground_truths)
    
    batch = {
        'raw_utterances': [lang],
        'initial_frames': [torch.from_numpy(img).cuda()],
        'ground_truths': [ground_truths],
        'program_lists': [utterance2program(lang, cliport=is_cliport)],
        'gt_place_boxes': [gt_place_boxes],
        'move_all': [move_all]
    }

    outputs, _, visualize_outputs, columns, video_dict = executor(batch)
    outputs = outputs[0]
    outputs = (
        outputs[0].float().round().long(),
        outputs[1].float().cpu().round().long()
    )
    pick_hmap = torch.zeros(len(outputs[0]), img.shape[0], img.shape[1])
    for k in range(len(pick_hmap)):
        _out = outputs[0][k]
        pick_hmap[k, _out[1]:_out[3], _out[0]:_out[2]] = 1
    if pick_hmap.sum() == 0:
        assert False, "pick_hmap is empty"

    place_hmap = torch.zeros(len(outputs[1]), img.shape[0], img.shape[1])
    for k in range(len(place_hmap)):
        _out = outputs[1][k]
        place_hmap[k, _out[1]:_out[3], _out[0]:_out[2]] = 1

    if place_hmap.sum() == 0:
        assert False, "place_hmap is empty"

    # flip pick_hmap and place_hmap from 320X640 to 640X320
    pick_hmap, place_hmap = flip_heatmaps(pick_hmap, place_hmap)

    assert pick_hmap.sum() != 0 and place_hmap.sum() != 0
    return {
        "pick_hmap": pick_hmap,
        "place_hmap": place_hmap,
        "visualize_outputs": visualize_outputs,
        "columns": columns,
        "pick_boxes": flip_boxes(outputs[0]),
        "place_boxes": flip_boxes(outputs[1]),
        "video_dict": video_dict
    }


def _visualize(img_color, depth, map_list=[], text=None):
    """Adapted from CLIPort, img (H, W, 3), conf (H, W)."""
    cmap = 'plasma'
    alpha = 0.8
    nrows = 1 + len(map_list)
    _, axs = plt.subplots(nrows, 2, figsize=(20, 10), squeeze=False)
    for i in range(nrows):
        for j in range(2):
            axs[i][j].axes.xaxis.set_visible(False)
            axs[i][j].axes.yaxis.set_visible(False)
            if (i, j) == (0, 1):
                axs[i][j].imshow(depth.T, cmap=cmap)
            else:
                axs[i][j].imshow(img_color.transpose(1, 0, 2) / 255)
            axs[i][j].axes.xaxis.set_visible(False)
            axs[i][j].axes.yaxis.set_visible(False)
    axs[0][0].set_title('Image')
    axs[0][1].set_title('Depth')
    if text is not None:
        axs[0][0].text(0.15, 0.15, text, dict(size=16))

    row_cnt = 1
    for item in map_list:
        pick = _visualize_map(item['pick'], item.get('type', 'none'))
        place = _visualize_map(item['place'], item.get('type', 'none'))
        alp = 1 if item.get('type', 'none') == 'img' else alpha
        axs[row_cnt][0].imshow(pick, cmap=cmap, alpha=alp)
        axs[row_cnt][0].set_title('Pick ' + item['name'])
        axs[row_cnt][1].imshow(place, cmap=cmap, alpha=alp)
        axs[row_cnt][1].set_title('Place ' + item['name'])
        row_cnt += 1

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = np.array(Image.open(buf))  # H x W x 4
    image = image[:, :, :3]
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image)  # 3 x H x W
    plt.close('all')
    return image


def _visualize_map(label_map, map_type='conf'):
    """Visualize a small box around a point."""
    if map_type == 'img':
        return label_map.transpose(1, 0, 2)
    scale = 255.0
    if map_type == 'point':
        (i, j) = np.where(label_map == 1)
        (i, j) = (i[0], j[0])
        label_map[max(i-5, 0):i+5, max(j-5, 0):j+5] = 1
    max_, min_ = label_map.max(), label_map.min()
    label_map = (label_map - min_) / (max_ - min_)
    label_map = np.uint8(label_map[..., None] * scale).transpose(1, 0, 2)
    label_map = np.ma.masked_where(label_map < 0, label_map)
    return label_map


def main():
    """Run main training/test pipeline."""
    data_path = "/projects/katefgroup/language_grounding/"
    if not osp.exists(data_path):
        data_path = '/home/yunchuz/ns_transporter'  # or change this if you work locally

    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--checkpoint_path", default=osp.join(data_path, "checkpoints/")
    )
    argparser.add_argument("--checkpoint", default="transporter.pt")
    argparser.add_argument("--epochs", default=150, type=int)
    argparser.add_argument("--batch_size", default=8, type=int)
    argparser.add_argument("--lr", default=1e-4, type=float)
    argparser.add_argument("--tensorboard_dir", default="transporter_debug")
    argparser.add_argument("--eval", action='store_true')
    argparser.add_argument("--record", action='store_true')
    argparser.add_argument("--n_rotations", default=36, type=int)
    argparser.add_argument("--multi_task", action='store_true')
    argparser.add_argument("--eval_task", default="place-red-in-green")
    argparser.add_argument("--eval_with_executor", action='store_true')
    argparser.add_argument("--eval_on_env", default=True, action='store_true')

    argparser.add_argument("--langevin_steps", default=30, type=int)
    argparser.add_argument("--ld_lr", default=10, type=int)
    argparser.add_argument("--data_root", default='ns_transporter_data/transporter_data_sep_100d_new', type=str)
    argparser.add_argument("--checkpoint_prefix", default='checkpoints')
    argparser.add_argument("--filter", default=True, type=bool)
    argparser.add_argument("--visualize", default=False, action='store_true')
    argparser.add_argument("--constant_bg", default=False, action='store_true')
    argparser.add_argument("--eval_list", default=None, nargs='+', type=int)
    argparser.add_argument("--overhead", default=False, action='store_true')
    argparser.add_argument("--verbose", default=False, action='store_true')
    argparser.add_argument("--eval_freq", default=3, type=int)
    argparser.add_argument("--pretrained", action='store_true')
    argparser.add_argument("--debug", action='store_true')
    argparser.add_argument("--oracle", action='store_true')
    argparser.add_argument("--ndemos_train", default=10, type=int)
    argparser.add_argument("--ndemos_test", default=50, type=int)
    argparser.add_argument("--ndemos_val", default=5, type=int)
    argparser.add_argument("--theta_sigma", default=60, type=int)
    argparser.add_argument("--eval_all", action='store_true')
    argparser.add_argument("--goal_conditioned_training", action='store_true')
    argparser.add_argument("--relations", action='store_true')
    argparser.add_argument("--multi_relations", action='store_true')
    argparser.add_argument("--multi_relations_group", action='store_true')
    argparser.add_argument("--shapes", action='store_true')
    argparser.add_argument("--skip_unseen", action='store_true')

    argparser = get_bdetr_args(argparser)
    args = argparser.parse_args()
    print(args)
    args.ckpnt = osp.join(args.checkpoint_path, args.checkpoint)
    print(args.ckpnt)

    # Other variables
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device
    os.makedirs(args.checkpoint_path, exist_ok=True)

    if args.relations:
        all_train_tasks = [
            'right-seen-colors',
            'above-seen-colors',
            'below-seen-colors',
            'left-seen-colors',
        ]

        all_eval_tasks = [
            'right-seen-colors',
            'above-seen-colors',
            'below-seen-colors',
            'left-seen-colors',
        ]
    elif args.multi_relations:
        all_train_tasks = [
            "composition-seen-colors"
        ]
        all_eval_tasks = [
            "composition-seen-colors"
        ]
    elif args.multi_relations_group:
        all_train_tasks = [
            "composition-seen-colors-group"
        ]
        all_eval_tasks = [
            "composition-seen-colors-group"
        ]
    elif args.shapes:
        all_train_tasks = [
            "circle-seen-colors",
            'line-seen-colors'
        ]
        all_eval_tasks = [
            "circle-seen-colors",
            'line-seen-colors'
        ]
    else:
        all_train_tasks = [
            'assembling-kits-seq-seen-colors',
            'packing-seen-google-objects-group',
            'packing-seen-google-objects-seq',
            'put-block-in-bowl-seen-colors'
        ]

        all_eval_tasks = [
            'assembling-kits-seq-seen-colors',
            'packing-seen-google-objects-group',
            'packing-seen-google-objects-seq',
            'put-block-in-bowl-seen-colors'
        ]

    unseen_tasks = [t.replace("seen", "unseen") for t in all_eval_tasks]
    if args.skip_unseen:
        unseen_tasks = []
    all_eval_tasks.extend(unseen_tasks)

    with open("cfg/train.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    # Loaders
    if not args.multi_task:
        datasets = {
            mode: RavensDataset(
                args.data_root,
                task_list=[args.eval_task],
            n_demos=args.ndemos_train if mode in {'train', 'val'} else args.ndemos_test,
                split=mode,
                augment=(mode == 'train' and not args.debug),
                theta_sigma=args.theta_sigma,
                cliport=(not args.relations and not args.multi_relations and not args.multi_relations_group and not args.shapes)
            )
            for mode in ('train', 'val', 'test')
        }
    else:
        datasets = {
            mode: RavensDataset(
                args.data_root,
                task_list=all_train_tasks if mode == 'train' else all_eval_tasks,
                n_demos=args.ndemos_train if mode == 'train' else args.ndemos_val,
                split=mode,
                augment=(mode == 'train' and not args.debug),
                theta_sigma=args.theta_sigma,
                cliport=(not args.relations and not args.multi_relations and not args.multi_relations_group and not args.shapes)
            )
            for mode in ('train', 'val')
        }
        datasets['test'] = RavensDataset(
            args.data_root,
            task_list=all_eval_tasks,
            n_demos=args.ndemos_test,
            split='test',
            augment=False,
            cliport=(not args.relations and not args.multi_relations and not args.multi_relations_group and not args.shapes)
        )

    if args.debug:
        print("In debugging mode")
        datasets["val"] = datasets["train"]
        datasets["test"] = datasets["train"]

    print(len(datasets['train']), len(datasets['test']))
    data_loaders = {
        mode: DataLoader(
            datasets[mode],
            batch_size=args.batch_size,
            shuffle=mode != 'test',
            drop_last=mode == 'train',
        )
        for mode in ('train', 'val', 'test')
    }

    # Models
    if args.goal_conditioned_training:
        model = GCTransporterAgent(args.n_rotations, cfg, args.pretrained)
    else:
        model = OriginalTransporterAgent(args.n_rotations, cfg, args.pretrained)
    trainer = Trainer(model.to(args.device), data_loaders, args)
    trainer.run()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
