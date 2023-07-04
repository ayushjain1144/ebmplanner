"""Pipeline for training/testing."""

import argparse
import os
import os.path as osp

import torch
from torch.utils.data import DataLoader

from ebm_training.utils.custom_datasets import EBMDataset


def main():
    """Run main training/test pipeline."""
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--checkpoint_path", default="checkpoints/")
    argparser.add_argument("--checkpoint", default="relation_ebm.pt")
    argparser.add_argument("--epochs", default=2, type=int)
    argparser.add_argument("--batch_size", default=128, type=int)
    argparser.add_argument("--lr", default=1e-4, type=float)
    argparser.add_argument("--tensorboard_dir", default="ebm")
    argparser.add_argument("--eval", action='store_true')
    argparser.add_argument("--langevin_steps", default=30, type=int)
    argparser.add_argument("--buffer_size", default=100000, type=int)
    argparser.add_argument("--step_lr", default=1.0, type=float)
    argparser.add_argument("--kl_coeff", default=1.0, type=float)
    argparser.add_argument("--use_buffer", action='store_true')
    argparser.add_argument("--use_disc", action='store_true')
    argparser.add_argument("--use_kl", action='store_true')
    argparser.add_argument("--concept", default="front")
    argparser.add_argument("--n_samples", default=20, type=int)
    argparser.add_argument("--skip", action='store_true')

    args = argparser.parse_args()
    if args.skip:
        return None
    args.ckpnt = osp.join(args.checkpoint_path, args.checkpoint)
    args.disable_val_grad = False

    # Other variables
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # Loaders
    concepts = [args.concept]
    datasets = {
        split: EBMDataset(split, concept=concepts, n_samples=args.n_samples)
        for split in ('train', 'val')
    }
    print(len(datasets['train']), len(datasets['val']))
    data_loaders = {
        mode: DataLoader(
            datasets[mode],
            batch_size=args.batch_size,
            shuffle=mode == 'train',
            drop_last=mode == 'train',
            num_workers=4
        )
        for mode in ('train', 'val')
    }

    # Models
    rel_list = ['front', 'behind', 'left', 'right', 'inside']
    shape_list = ['line', 'circle', 'triangle', 'square']
    rel3d_list = ['supported-by']
    pose_list = ['racer']
    if args.concept in rel_list:
        from ebm_training.ebm_models import RelationEBM
        from ebm_training.ebm_trainers import RelationEBMTrainer
        from ebm_training.classifiers import CLASSIFIERS
        model = RelationEBM()
        trainer = RelationEBMTrainer(
            model.to(args.device), data_loaders, args,
            CLASSIFIERS[args.concept]
        )
    elif args.concept in shape_list:
        from ebm_training.ebm_models import ShapeEBM
        from ebm_training.ebm_trainers import ShapeEBMTrainer
        model = ShapeEBM()
        trainer = ShapeEBMTrainer(
            model.to(args.device), data_loaders, args, None
        )
    elif args.concept in rel3d_list:
        from ebm_training.ebm_models import RelationEBM3D
        from ebm_training.ebm_trainers import Relation3DEBMTrainer
        model = RelationEBM3D()
        trainer = Relation3DEBMTrainer(
            model.to(args.device), data_loaders, args, None
        )
    elif args.concept in pose_list:
        from ebm_training.ebm_models import PosedShapeEBM
        from ebm_training.ebm_trainers import PosedShapeEBMTrainer
        model = PosedShapeEBM()
        trainer = PosedShapeEBMTrainer(
            model.to(args.device), data_loaders, args, None
        )
    trainer.run()


if __name__ == "__main__":
    main()
