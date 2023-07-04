"""Pipeline for parser training/testing."""

import argparse
import os
import os.path as osp

import pkbar
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.parser_dataset import ParserDataset, parser_collate_fn
from models.parser_cliport import Seq2TreeTransformer

import ipdb
st = ipdb.set_trace


def train_parser(model, data_loaders, args):
    """Train a language-to-program parser."""
    # Set
    writer = SummaryWriter(f'runs/{args.tensorboard_dir}')
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    start_epoch = 0
    # st()
    # Load
    if osp.exists(args.parser_ckpnt):
        checkpoint = torch.load(args.parser_ckpnt)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]

    # Eval?
    if args.eval:
        model.eval()
        test_acc = eval_parser(model, data_loaders['test'], args)
        print(f"Test Accuracy: {test_acc}")
        return model

    # Go!
    val_acc_prev_best = -1.0
    for epoch in range(start_epoch, args.epochs):
        print("Epoch: %d/%d" % (epoch + 1, args.epochs))
        kbar = pkbar.Kbar(target=len(data_loaders['train']), width=25)
        model.train()
        # Train
        for step, ex in enumerate(data_loaders['train']):
            loss, _ = model(ex["raw_utterances"], ex["program_trees"])
            kbar.update(step, [("loss", loss)])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Print loss
            writer.add_scalar(
                'training_loss', loss.item(),
                epoch * len(data_loaders['train']) + step
            )

        # Validate
        val_acc = 0.0
        if epoch > 15:
            print("\nValidation")
            val_acc = eval_parser(model, data_loaders['val'], args)
            writer.add_scalar("val_acc", val_acc, epoch)

        # Store
        if epoch <= 2 or val_acc >= val_acc_prev_best:
            print("Saving Checkpoint")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                args.parser_ckpnt
            )
            val_acc_prev_best = val_acc

    # Test
    model.eval()
    test_acc = eval_parser(model, data_loaders['test'], args)
    print(f"Test Accuracy: {test_acc}")
    return model


@torch.no_grad()
def eval_parser(model, data_loader, args):
    """Evaluate model on val/test data."""
    model.eval()
    kbar = pkbar.Kbar(target=len(data_loader), width=25)
    num_correct = 0
    num_examples = 0
    val_accumulated = 0.0
    debug = False
    val_acc = 0
    for step, ex in enumerate(data_loader):
        try:
            _, progs = model(
                ex["raw_utterances"], ex['program_trees'], 
                teacher_forcing=False, compute_loss=False
            )
            num_correct += sum(
                int(progs[i] == ex["program_lists"][i])
                for i in range(len(progs))
            )

            if debug:
                for i in range(len(progs)):
                    if progs[i] != ex["program_lists"][i]:
                        print(f"Raw utterance: {ex['raw_utterances'][i]}")
                        print(f"gt_prog: {ex['program_lists'][i]}\n")
                        print(f"predicted_prog: {progs[i]}")
                        

            num_examples += len(progs)
            kbar.update(step, [("accuracy", num_correct / num_examples)])
            val_acc = num_correct / num_examples
            val_accumulated += val_acc
        except RecursionError as _:
            print("Recursion depth exceeded while evaluation")
            continue

    print(f"\nAccuracy: {val_accumulated/len(data_loader)}")
    return val_acc


def main():
    """Run main training/test pipeline."""
    data_path = "/projects/katefgroup/language_grounding/"
    if not osp.exists(data_path):
        data_path = 'data/'  # or change this if you work locally

    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--checkpoint_path", default=osp.join(data_path, "checkpoints/")
    )
    argparser.add_argument("--checkpoint", default="parser_cliport_sep.pt")
    argparser.add_argument("--epochs", default=512, type=int)
    argparser.add_argument("--batch_size", default=32, type=int)
    argparser.add_argument("--lr", default=1e-3, type=float)
    argparser.add_argument("--wd", default=1e-7, type=float)
    argparser.add_argument("--tensorboard_dir", default="exp1")
    argparser.add_argument("--eval", action='store_true')
    argparser.add_argument("--annos_path", default='data/cliport_program_annos.json', type=str)

    args = argparser.parse_args()
    args.parser_ckpnt = osp.join(args.checkpoint_path, args.checkpoint)

    # Other variables
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # Parser
    parser_datasets = {
        mode: ParserDataset(split=mode, annos_path=args.annos_path)
        for mode in ('train', 'val', 'test')
    }
    data_loaders = {
        mode: DataLoader(
            parser_datasets[mode],
            batch_size=args.batch_size,
            collate_fn=parser_collate_fn,
            shuffle=mode == 'train',
            drop_last=mode == 'train',
            num_workers=0
        )
        for mode in ['train', 'val', 'test']
    }

    # data_loaders['test'] = data_loaders['val']
    train_parser(  # tests also
        Seq2TreeTransformer().to(device),
        data_loaders, args
    )


if __name__ == "__main__":
    main()
