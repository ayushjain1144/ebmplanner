from copy import deepcopy
import os.path as osp

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .utils.misc import Summ_writer
from .utils.replay_buffer import ReplayBuffer
from .utils.visualization import (
    plot_relations_2d,
    plot_boxes_2d_all,
    plot_relations_3d,
    plot_pose
)


class EBMTrainer:
    """Train/test models."""

    def __init__(self, model, data_loaders, args, classifier=None):
        self.model = model
        self.data_loaders = data_loaders
        self.args = args
        self.classifier = classifier

        self.writer = SummaryWriter(f'runs/{args.tensorboard_dir}')
        self.vis_cnt = 0
        self.optimizer = Adam(
            model.parameters(), lr=args.lr, betas=(0.0, 0.9), eps=1e-8
        )
        self.buffer = []
        if args.use_buffer:
            self.buffer = ReplayBuffer(args.buffer_size)

    def run(self):
        # Set
        start_epoch = 0
        val_acc_prev_best = -1.0

        # Load
        if osp.exists(self.args.ckpnt):
            start_epoch, val_acc_prev_best = self._load_ckpnt()

        # Eval?
        if self.args.eval or start_epoch >= self.args.epochs:
            self.model.eval()
            self.train_test_loop('val')
            return self.model

        # Go!
        for epoch in range(start_epoch, self.args.epochs):
            print("Epoch: %d/%d" % (epoch + 1, self.args.epochs))
            self.model.train()
            # Train
            self.train_test_loop('train', epoch)
            # Validate
            print("\nValidation")
            self.model.eval()
            with torch.set_grad_enabled(not self.args.disable_val_grad):
                val_acc = self.train_test_loop('val', epoch)

            # Store
            if val_acc >= val_acc_prev_best:
                print("Saving Checkpoint")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_acc": val_acc
                }, self.args.ckpnt)
                val_acc_prev_best = val_acc
            else:
                print("Updating Checkpoint")
                checkpoint = torch.load(self.args.ckpnt)
                checkpoint["epoch"] += 1
                torch.save(checkpoint, self.args.ckpnt)

        self.writer.close()
        return self.model

    def _load_ckpnt(self):
        ckpnt = torch.load(self.args.ckpnt)
        self.model.load_state_dict(ckpnt["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(ckpnt["optimizer_state_dict"])
        start_epoch = ckpnt["epoch"]
        val_acc_prev_best = ckpnt['best_acc']
        return start_epoch, val_acc_prev_best

    def _prepare_inputs(self, batch):
        return {'pos': batch, 'neg': batch}

    def _sample_from_buffer(self, inputs):
        replay_batch = self.buffer.sample(len(inputs['pos'][0]))
        replay_mask = np.random.uniform(0, 1, len(replay_batch[0])) > 0.7
        for k in range(len(replay_batch)):  # loop over fields
            if isinstance(inputs['neg'][k], list):
                inputs['neg'][k] = np.asarray(inputs['neg'][k])
                inputs['neg'][k][replay_mask] = \
                    np.asarray(replay_batch[k])[replay_mask]
                inputs['neg'][k] = inputs['neg'][k].tolist()
            else:
                device = inputs['neg'][k].device
                inputs['neg'][k][replay_mask] = \
                    replay_batch[k][replay_mask].to(device)
        return inputs

    def train_test_loop(self, mode='train', epoch=1000):
        acc, samples, gt_acc = 0, 1e-14, 0
        for step, ex in tqdm(enumerate(self.data_loaders[mode])):
            inputs = self._prepare_inputs(ex)
            inputs['noisy'] = deepcopy(inputs['neg'])

            # Load from buffer
            if len(self.buffer) > self.args.batch_size and mode == 'train':
                inputs = self._sample_from_buffer(inputs)

            # Run Langevin dynamics
            neg, neg_kl, seq = self.langevin(inputs['neg'])

            # Save to buffer
            if self.args.use_buffer and mode == 'train':
                self.buffer.add(neg)

            # Compute energies
            energy_pos = self.model(inputs['pos'])
            energy_neg = self.model(neg)

            # Losses
            loss = energy_pos.mean() - energy_neg.mean()
            loss = loss + ((energy_pos ** 2).mean() + (energy_neg ** 2).mean())
            if self.args.use_kl:
                self.model.requires_grad_(False)
                loss_kl = self.model(neg_kl).mean()
                self.model.requires_grad_(True)
            else:
                loss_kl = 0
            loss = loss + self.args.kl_coeff * loss_kl

            if self.classifier is not None:
                with torch.no_grad():
                    score_pos = self.classifier(neg)
                    acc += (score_pos > 0).sum().item()
                    samples += len(score_pos)
                    score_pos = self.classifier(inputs['pos'])
                    gt_acc += (score_pos > 0).sum().item()

            # Update
            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Logging
            self.writer.add_scalar(
                'Positive_energy_avg/' + mode, energy_pos.mean().item(),
                epoch * len(self.data_loaders[mode]) + step
            )
            self.writer.add_scalar(
                'Negative_energy_avg/' + mode, energy_neg.mean().item(),
                epoch * len(self.data_loaders[mode]) + step
            )
            self.writer.add_scalar(
                'Energy_diff/' + mode,
                abs(energy_pos.mean().item() - energy_neg.mean().item()),
                epoch * len(self.data_loaders[mode]) + step
            )

            # Visualizations
            if self.args.eval and (step + 1) % 30 == 0:
                self._visualize(seq, self.vis_cnt)
                self.vis_cnt += 1
        print(acc / samples, gt_acc / samples)
        return acc / samples

    def langevin(self, neg):
        """Langevin dynamics implemented for boxes and relations."""
        sbox_centers, sbox_sizes, obox_centers, obox_sizes, rels = neg
        noise = torch.randn_like(sbox_centers).detach()
        negs_samples = []

        for i in range(self.args.langevin_steps):
            # Add noise
            noise.normal_()
            sbox_centers = sbox_centers + 0.005 * noise

            # Forward pass
            sbox_centers.requires_grad_(requires_grad=True)
            energy = self.model((
                sbox_centers, sbox_sizes, obox_centers, obox_sizes, rels
            ))

            # Backward pass (gradients wrt image)
            _grad = torch.autograd.grad([energy.sum()], [sbox_centers])[0]
            sbox_centers_kl = sbox_centers.clone()
            sbox_centers = sbox_centers - self.args.step_lr * _grad

            # Compute kl image for last step
            if i == self.args.langevin_steps - 1:
                energy = self.model((
                    sbox_centers_kl, sbox_sizes, obox_centers, obox_sizes, rels
                ))
                _grad = torch.autograd.grad(
                    [energy.sum()], [sbox_centers_kl],
                    create_graph=True
                )[0]
                sbox_centers_kl = sbox_centers_kl - self.args.step_lr * _grad
                sbox_centers_kl = torch.clamp(sbox_centers_kl, -1, 1)

            # Detach/clamp/store
            sbox_centers = torch.clamp(sbox_centers.detach(), -1, 1)
            negs_samples.append((
                sbox_centers, sbox_sizes, obox_centers, obox_sizes, rels
            ))

        return (
            (sbox_centers, sbox_sizes, obox_centers, obox_sizes, rels),
            (sbox_centers_kl, sbox_sizes, obox_centers, obox_sizes, rels),
            negs_samples
        )

    def _visualize(self, seq, vis_cnt):
        pass


class RelationEBMTrainer(EBMTrainer):
    """Train/test relation EBM with boxes."""

    def __init__(self, model, data_loaders, args, classifier=None):
        super().__init__(model, data_loaders, args, classifier)

    def _prepare_inputs(self, batch):
        device = self.args.device
        return {
            'pos': (
                batch['sboxes'][..., :2].float().to(device),
                batch['sboxes'][..., 2:].float().to(device),
                batch['oboxes'][..., :2].float().to(device),
                batch['oboxes'][..., 2:].float().to(device),
                batch['label'].to(device)
            ),
            'neg': (
                batch['noisy_sboxes'][..., :2].float().to(device),
                batch['noisy_sboxes'][..., 2:].float().to(device),
                batch['oboxes'][..., :2].float().to(device),
                batch['oboxes'][..., 2:].float().to(device),
                batch['label'].to(device)
            )
        }

    def _visualize(self, seq, vis_cnt):
        sw = Summ_writer(
            writer=self.writer,
            global_step=vis_cnt,
            log_freq=1,
            fps=2
        )
        rel_names = self.data_loaders['train'].dataset.rel_list
        vis_neg_ld_images = torch.stack([
            plot_relations_2d(
                torch.cat((s[0][0], s[1][0])).detach().cpu().numpy(),
                torch.cat((s[2][0], s[3][0])).detach().cpu().numpy()
            )
            for s in seq[::3]
        ], dim=1)
        sw.summ_gif(
            f'{rel_names[seq[0][4][0]]}/start_to_goal_gif', vis_neg_ld_images
        )


class ShapeEBMTrainer(EBMTrainer):
    """Train/test shape EBM with boxes."""

    def __init__(self, model, data_loaders, args, classifier=None):
        super().__init__(model, data_loaders, args, classifier)

    def langevin(self, neg):
        """Langevin dynamics implemented for boxes and relations."""
        sbox_centers, sbox_sizes, rel, mask = neg
        noise = torch.randn_like(sbox_centers).detach()
        negs_samples = []

        for i in range(self.args.langevin_steps):
            # Add noise
            noise.normal_()
            sbox_centers = sbox_centers + 0.005 * noise

            # Forward pass
            sbox_centers.requires_grad_(requires_grad=True)
            energy = self.model((sbox_centers, sbox_sizes, rel, mask))

            # Backward pass (gradients wrt image)
            _grad = torch.autograd.grad([energy.sum()], [sbox_centers])[0]
            sbox_centers_kl = sbox_centers.clone()
            sbox_centers = sbox_centers - self.args.step_lr * _grad

            # Compute kl image for last step
            if i == self.args.langevin_steps - 1:
                energy = self.model((sbox_centers_kl, sbox_sizes, rel, mask))
                _grad = torch.autograd.grad(
                    [energy.sum()], [sbox_centers_kl],
                    create_graph=True
                )[0]
                sbox_centers_kl = sbox_centers_kl - self.args.step_lr * _grad
                sbox_centers_kl = torch.clamp(sbox_centers_kl, -1, 1)

            # Detach/clamp/store
            sbox_centers = torch.clamp(sbox_centers.detach(), -1, 1)
            negs_samples.append((sbox_centers, sbox_sizes, rel, mask))

        return (
            (sbox_centers, sbox_sizes, rel, mask),
            (sbox_centers_kl, sbox_sizes, rel, mask),
            negs_samples
        )

    def _prepare_inputs(self, batch):
        device = self.args.device
        return {
            'pos': (
                batch['sboxes'][..., :2].float().to(device),
                batch['sboxes'][..., 2:].float().to(device),
                batch['label'].long().to(device),
                batch['attention'].long().to(device)
            ),
            'neg': (
                batch['noisy_sboxes'][..., :2].float().to(device),
                batch['noisy_sboxes'][..., 2:].float().to(device),
                batch['label'].long().to(device),
                batch['attention'].long().to(device)
            )
        }

    def _visualize(self, seq, vis_cnt):
        sw = Summ_writer(
            writer=self.writer,
            global_step=vis_cnt,
            log_freq=1,
            fps=2
        )
        vis_neg_ld_images = torch.stack([
            plot_boxes_2d_all(
                torch.ones(len(s[0][0])).numpy(),
                torch.cat((s[0][0], s[1][0]), -1).detach().cpu().numpy()
            )
            for s in seq[::3]
        ], dim=1)
        sw.summ_gif('start_to_goal_gif', vis_neg_ld_images)


class Relation3DEBMTrainer(EBMTrainer):
    """Train/test relation EBM with 3D boxes."""

    def __init__(self, model, data_loaders, args, classifier=None):
        super().__init__(model, data_loaders, args, classifier)

    def _prepare_inputs(self, batch):
        device = self.args.device
        return {
            'pos': (
                batch['sboxes'][..., :3].float().to(device),
                batch['sboxes'][..., 3:].float().to(device),
                batch['oboxes'][..., :3].float().to(device),
                batch['oboxes'][..., 3:].float().to(device),
                batch['label'].to(device)
            ),
            'neg': (
                batch['noisy_sboxes'][..., :3].float().to(device),
                batch['noisy_sboxes'][..., 3:].float().to(device),
                batch['oboxes'][..., :3].float().to(device),
                batch['oboxes'][..., 3:].float().to(device),
                batch['label'].to(device)
            )
        }

    def _visualize(self, seq, vis_cnt):
        sw = Summ_writer(
            writer=self.writer,
            global_step=vis_cnt,
            log_freq=1,
            fps=2
        )
        rel_names = self.data_loaders['train'].dataset.rel_list
        vis_neg_ld_images = torch.stack([
            plot_relations_3d(
                torch.cat((s[0][0], s[1][0])).detach().cpu().numpy(),
                torch.cat((s[2][0], s[3][0])).detach().cpu().numpy()
            )
            for s in seq[::3]
        ], dim=1)
        sw.summ_gif(
            f'{rel_names[seq[0][4][0]]}/start_to_goal_gif', vis_neg_ld_images
        )


class PosedShapeEBMTrainer(EBMTrainer):
    """Train/test relation EBM with posed boxes."""

    def __init__(self, model, data_loaders, args, classifier=None):
        super().__init__(model, data_loaders, args, classifier)

    def langevin(self, neg):
        """Langevin dynamics implemented for boxes and relations."""
        points, theta = neg
        neg = torch.cat((points, theta[:, :, None]), -1)
        noise = torch.randn_like(neg).detach()
        negs_samples = []

        for i in range(self.args.langevin_steps):
            # Add noise
            noise.normal_()
            neg = neg + 0.005 * noise

            # Forward pass
            neg.requires_grad_(requires_grad=True)
            energy = self.model((neg[..., :2], neg[..., -1]))

            # Backward pass (gradients wrt image)
            _grad = torch.autograd.grad([energy.sum()], [neg])[0]
            neg_kl = neg.clone()
            neg = neg - self.args.step_lr * _grad

            # Compute kl image for last step
            if i == self.args.langevin_steps - 1:
                energy = self.model((neg_kl[..., :2], neg_kl[..., -1]))
                _grad = torch.autograd.grad(
                    [energy.sum()], [neg_kl],
                    create_graph=True
                )[0]
                neg_kl = neg_kl - self.args.step_lr * _grad
                neg_kl = torch.clamp(neg_kl, -1, 1)

            # Detach/clamp/store
            theta = torch.clamp(theta.detach(), -1, 1)
            negs_samples.append((neg[..., :2], neg[..., -1]))

        return (
            (neg[..., :2], neg[..., -1]),
            (neg_kl[..., :2], neg_kl[..., -1]),
            negs_samples
        )

    def _prepare_inputs(self, batch):
        device = self.args.device
        return {
            'pos': (
                batch['points'].float().to(device),
                batch['theta'].float().to(device)
            ),
            'neg': (
                batch['neg_points'].float().to(device),
                batch['neg_theta'].float().to(device)
            )
        }

    def _visualize(self, seq, vis_cnt):
        sw = Summ_writer(
            writer=self.writer,
            global_step=vis_cnt,
            log_freq=1,
            fps=2
        )
        vis_neg_ld_images = torch.stack([
            plot_pose(
                s[0][0].detach().cpu().numpy(),
                np.pi * s[1][0].detach().cpu().numpy(),
                np.zeros((len(s[0][0]), 2)),
                name=None,
                show_line=False,
                show_center=False,
                show_target=False,
                return_buf=True
            )
            for s in seq[::3]
        ], dim=1)
        sw.summ_gif('start_to_goal_gif', vis_neg_ld_images)


def clip_grad_norm_(parameters, max_norm, norm_type=2.0):
    """
    Clip gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int or str): type of the used p-norm.
            Can be ``'inf'`` for infinity norm.
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == 'inf':
        total_norm = max(
            p.grad.detach().abs().max().to(device) for p in parameters
        )
    else:
        total_norm = torch.norm(torch.stack([
            torch.norm(p.grad.detach(), norm_type).to(device)
            for p in parameters
        ]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm
