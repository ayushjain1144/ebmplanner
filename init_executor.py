import torch

from models.ebms import RelationEBM, ShapeEBM
from models.parser_cliport import Seq2TreeTransformer
from models.ns_transporter import NS_Transporter
from beauty_detr import build_bdetr_model


def _load_ebm(model, ckpt, device):
    ebm = model.to(device)
    checkpoint = torch.load(ckpt, map_location=device)
    ebm.load_state_dict(checkpoint["model_state_dict"], strict=False)
    ebm.eval()
    for param in ebm.parameters():
        param.requires_grad_(False)
    return ebm


def _initialize_executor(args):
    parser = Seq2TreeTransformer().to(args.device)
    # beauty detr checkppoint
    if args.relations or args.multi_relations or args.multi_relations_group:
        type = "relations"
    elif args.shapes:
        type = "shapes"
    else:
        type = "cliport"

    checkpoint_path = f"{args.checkpoint_prefix}/parser_{type}_{args.ndemos_train}.pt"
    print(f"Loading parser checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    parser.load_state_dict(checkpoint["model_state_dict"], strict=False)
    parser.eval()

    # Load beauty-detr
    bdetr_model, _, _ = \
        build_bdetr_model(args)
    bdetr_model.to(args.device)

    checkpoint_path = f"{args.checkpoint_prefix}/bdetr_{type}_{args.ndemos_train}.pt"
    print(f"Loading beauty-detr checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    bdetr_model.load_state_dict(checkpoint["model"], strict=False)
    bdetr_model.eval()

    # load ebms
    device = args.device
    ebm_dict = {
        'circle': _load_ebm(ShapeEBM(), f"{args.checkpoint_prefix}/circle_10.pt", device),
        'line': _load_ebm(ShapeEBM(), f"{args.checkpoint_prefix}/line_10.pt", device),
        'inside': _load_ebm(RelationEBM(), f"{args.checkpoint_prefix}/inside_10.pt", device),
        'left':  _load_ebm(RelationEBM(), f"{args.checkpoint_prefix}/left_10.pt", device),
        'right': _load_ebm(RelationEBM(), f"{args.checkpoint_prefix}/right_10.pt", device),
        'above': _load_ebm(RelationEBM(), f"{args.checkpoint_prefix}/front_10.pt", device),
        'below': _load_ebm(RelationEBM(), f"{args.checkpoint_prefix}/behind_10.pt", device)
    }
    ns_transporter = NS_Transporter(
        args,
        parser,
        bdetr_model,
        ebm_dict,
        visualize=args.visualize,
        verbose=args.verbose
    )
    return ns_transporter
