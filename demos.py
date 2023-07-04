"""Data collection script."""

import os
import pickle
import random

import numpy as np
import wandb

import tasks
from environments.environment import Environment
import ipdb
st = ipdb.set_trace


def main(cfg, env):
    # Set task
    task = tasks.names[cfg['task']]()
    task.mode = cfg['mode']
    task.name = cfg['task']
    env.set_task(task)
    # st()
    record = cfg['record']['save_video']
    save_data = cfg['save_data']  # if False, just loop for debugging
    force_generate = cfg['force_generate']

    # Initialize scripted oracle agent
    agent = task.oracle(env)
    data_path = os.path.join(
        cfg['data_dir'], "{}-{}".format(cfg['task'], task.mode)
    )
    print(f"Saving to: {data_path}")
    os.makedirs(data_path, exist_ok=True)

    # Train seeds are even and val/test seeds are odd.
    # Test seeds are offset by 10000
    seed, n_episodes = _get_seed_and_episodes(data_path, force_generate)
    if seed < 0 or not save_data:
        seed = _get_init_seed(task.mode)
        n_episodes = 0

    # Collect training data from oracle demonstrations.
    lang_goals_video = []
    while n_episodes < cfg['n_demos']:
        episode, total_reward = [], 0
        seed += 2

        # Set seeds.
        np.random.seed(seed)
        random.seed(seed)

        print('Oracle demo: {}/{} | Seed: {}'.format(
            n_episodes + 1, cfg['n_demos'], seed))
        env.set_task(task)

        obs = env.reset()

        if env.failed_datagen:
            env.set_failed_dategen(False)
            continue

        info = env.info
        reward = 0
        lang_goals_video.append(info['lang_goal'])

        # Start video recording (NOTE: super slow)
        if record:
            env.start_rec(f'{n_episodes+1:06d}')

        # Rollout expert policy
        for i in range(task.max_steps):
            # target reward for a successful action
            goal_reward = task.goals[0][-1]
            expected_steps = len(task.goals[0][0])
            target_reward = goal_reward / expected_steps - 0.003
            done_multitask = (i == task.max_steps - 1)
            # act
            act = agent.act(obs, info)
            if act is None:
                break
            episode.append((obs, act, reward, info))
            lang_goal = info['lang_goal']

            # take a step
            obs, _, _, _ = env.step(act)
            
            _, _, obj_mask = task.get_true_image(env)
            
            reward = env.task.reward(
                oracle=True, datagen=True, done_multitask=done_multitask, 
                obj_mask=obj_mask)
            done = env.task.done()
            info = env.info

            # did we hit the target reward?
            if reward < target_reward and cfg['discard_imperfect']:
                print(f"discraded: {reward} is not max: {target_reward}")
                break

            # we hit, so update reward
            total_reward += reward
            print(
                f'Total Reward: {total_reward:.3f}'
                f' | Done: {done} | Goal: {lang_goal}'
            )
            if done:
                break
        episode.append((obs, None, reward, info))

        # End video recording
        if record:
            env.end_rec()

        # Only save completed demonstrations.
        if save_data and total_reward > 0.99:
            _store_demo(data_path, seed, episode, n_episodes)
            n_episodes += 1
        elif total_reward <= 0.99:
            print("discarded")
            lang_goals_video = lang_goals_video[:-1]
        else:  # reward is ok but don't save
            n_episodes += 1

    # Play videos in wandb
    if record:
        _show_videos(cfg, lang_goals_video)


def _init_env(cfg):
    return Environment(
        cfg['assets_root'],
        disp=cfg['disp'],
        shared_memory=cfg['shared_memory'],
        hz=480,
        record_cfg=cfg['record'],
        debug=False,
        constant_bg=True
    )


def _get_seed_and_episodes(data_path, force_generate=False):
    color_path = os.path.join(data_path, 'action')
    n_episodes = 0
    max_seed = -1
    if os.path.exists(color_path) and not force_generate:
        for fname in sorted(os.listdir(color_path)):
            if '.pkl' in fname:
                seed = int(fname[(fname.find('-') + 1):-4])
                n_episodes += 1
                max_seed = max(max_seed, seed)
    else:
        print(f"{color_path} doesn't exist")
    return max_seed, n_episodes


def _get_init_seed(mode):
    if mode == 'train':
        seed = -2
    elif mode == 'val':
        seed = -1
    elif mode == 'test':
        seed = -1 + 10000
    else:
        raise Exception("Invalid mode. Valid options: train, val, test")
    return seed


def _dump(data, field, data_path, seed, n_episodes):
    field_path = os.path.join(data_path, field)
    os.makedirs(field_path, exist_ok=True)
    fname = f'{n_episodes:06d}-{seed}.pkl'  # -{len(episode):06d}
    with open(os.path.join(field_path, fname), 'wb') as f:
        pickle.dump(data, f)


def _store_demo(data_path, seed, episode, n_episodes):
    color, depth, action, reward, info = [], [], [], [], []
    for obs, act, r, i in episode:
        color.append(obs['color'])
        depth.append(obs['depth'])
        action.append(act)
        reward.append(r)
        info.append(i)
    color = np.uint8(color)
    depth = np.float32(depth)
    _dump(color, 'color', data_path, seed, n_episodes)
    _dump(depth, 'depth', data_path, seed, n_episodes)
    _dump(action, 'action', data_path, seed, n_episodes)
    _dump(reward, 'reward', data_path, seed, n_episodes)
    _dump(info, 'info', data_path, seed, n_episodes)


def _show_videos(cfg, lang_goals_video):
    folder = cfg['record']['save_video_path']
    for subdir, _, files in os.walk(folder):
        for i, file_ in enumerate(sorted(files)):
            if file_.endswith('mp4'):
                print(file_)
                video = wandb.Video(
                    os.path.join(folder, subdir, file_),
                    fps=25,
                    format="mp4",
                    caption=lang_goals_video[i]
                )
                wandb.log({"vis": video})


if __name__ == '__main__':
    cfg = {
        'assets_root': 'environments/assets/',
        'data_dir': 'benchmark_data/',
        'discard_imperfect': True,
        'disp': False,
        'record': {
            'save_video': False,
            'save_video_path':
                'benchmark_data/',
            'add_text': False,
            'fps': 25,
            'video_height': 640,
            'video_width': 720,
        },
        'save_data': True,
        'shared_memory': False,
        'force_generate': False ,
    }
    env = _init_env(cfg)
    task_list = [

        # cliport tasks
        'assembling-kits-seq-seen-colors',
        'packing-seen-google-objects-group',
        'packing-seen-google-objects-seq',
        'put-block-in-bowl-seen-colors',
        'assembling-kits-seq-unseen-colors',
        'packing-unseen-google-objects-group',
        'packing-unseen-google-objects-seq',
        'put-block-in-bowl-unseen-colors',

        # spatial relations
        'right-seen-colors',
        'above-seen-colors',
        'below-seen-colors',
        'left-seen-colors',
        'right-unseen-colors',
        'above-unseen-colors',
        'below-unseen-colors',
        'left-unseen-colors',

        #composition relations
        'composition-seen-colors',
        'composition-unseen-colors',
        'composition-seen-colors-group',
        'composition-unseen-colors-group'

        # shape
        'circle-seen-colors',
        'line-seen-colors',
        'circle-unseen-colors',
        'line-unseen-colors'
    ]
    splits = ['train', 'val', 'test']
    # splits = ['test']
    if cfg['record']['save_video']:
        splits = ['train']
        cfg['save_data'] = False
        wandb.init(project="robot", name="data_vis")
    for name in task_list:
        for split in splits:
            if split == "val":
                n_demos = 50
            elif split == "test":
                n_demos = 100
            else:
                n_demos = 100
            n_demos = 50 if split == 'val' else 100
            if cfg['record']['save_video']:
                n_demos = 2
            cfg['task'] = name
            cfg['mode'] = split
            cfg['n_demos'] = n_demos
            main(cfg, env)
