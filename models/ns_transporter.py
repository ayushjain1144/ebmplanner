import math

import numpy as np
import torch
from torch import nn

from data.create_cliport_programs import merge_programs
from beauty_detr.bdetr_eval_utils import transform, rescale_bboxes
from models.langevin_dynamics import langevin
from utils.executor_utils import clamp_boxes_torch

import wandb


XMIN = -1
XMAX = 1
YMIN = -0.5
YMAX = 0.5
ZMIN = -1
ZMAX = 1
SIZE = 0.15

LENS = torch.tensor(
    [XMAX - XMIN - SIZE, YMAX - YMIN - SIZE, ZMAX - ZMIN - SIZE]
) / 2.0


class NS_Transporter(nn.Module):
    """General class for symbolic transporter"""

    def __init__(
            self,
            args,
            parser,
            bdetr_model,
            ebm_dict,
            visualize=False,
            verbose=False
    ):
        """Initialize the modules"""
        super().__init__()
        self.parser = parser
        self.bdetr_model = bdetr_model
        self.ebm_dict = ebm_dict
        self.visualize = visualize
        self.verbose = verbose

        # this is a list of shape models
        self.args = args
        self.device = args.device

        self.score_thresh = 0.5

        # if self.visualize:
        wandb.init(project="NS_Transporter", name="2d_vis")

        self.bad_frames = []

        # ebm bounds are of size (2, 1) while
        # robots's bounds are from (1, 0.5)
        self.robot_bounds_to_ebm_bounds = 2.0

    def forward(self, batch):
        """Forward Pass."""
        module_outputs = {}

        # Run Parser
        programs = self._get_programs(
            batch['raw_utterances'], use_gt=False,
            gt_programs=batch.get('program_lists', None),
        )
        if self.verbose:
            print(batch['raw_utterances'])
            print(programs)
        module_outputs['pred_programs'] = programs

        batch_outputs = []

        # Run executors, seperately on each program
        for p, program in enumerate(programs):
            outputs = []
            img = batch['initial_frames'][p]
            ground_truth = batch['ground_truths'][p].copy() if batch['ground_truths'] is not None else None
            visualize_outputs = []
            columns = []
            video_dict = {
                "captions": [],
                "boxes": [],
                "ebm_boxes": [],
                "ebm_captions": []
            }

            for i, op in enumerate(program):
                if op['op'] == 'detect_objects':
                    outputs.append([])
                    continue
                elif op['op'] == 'filter':
                    if self.verbose:
                        print('filter')
                    # run clip here
                    class_label, location = op['concept']

                    # need to merge class label and location
                    # now that we use bdetr
                    assert location == "none", print(location)
                    predictions = self._filter(
                        img, class_label, ground_truths=ground_truth,
                        use_gt=False
                    )
                    if ground_truth is not None and len(ground_truth) > 0:
                        ground_truth.pop(0)
                    outputs.append(predictions)
                    if self.visualize:
                        visualize_outputs.append(
                            self._visualize(
                                img, predictions,
                                caption=f"filter {class_label}, location: {location}",
                                concept='filter'
                            )
                        )
                        columns.append(f'filter_{class_label}_{i}')

                    video_dict['captions'].append(f"{class_label}")
                    video_dict['boxes'].append(predictions.cpu().numpy())

                elif op['op'] == 'binaryEBM':
                    concepts = []
                    picks = []
                    places = []
                    caption = ""
                    for j in range(i, len(program)):
                        op = program[j]
                        concept_, _ = op['concept']
                        pick, place = op['inputs']
                        concepts.append(concept_)
                        caption += f"EBM({concept_}, {video_dict['captions'][pick-1]}, {video_dict['captions'][place-1]}) AND "
                        picks.append(pick)
                        places.append(place)
                    caption = caption[:-5]
                    video_dict['ebm_captions'].append(caption)
                    height, width = img.shape[:2]

                    boxes = []
                    count = 0
                    picks_ = []
                    places_ = []
                    concepts_ = []
                    done_idx = []
                    done_ebm = []
                    move_all = batch['move_all'][p]

                    for i, pick in enumerate(picks):
                        place = places[i]
                        pick_boxes_ = outputs[pick]
                        place_boxes_ = outputs[place]

                        if move_all:
                            pick_boxes_ = pick_boxes_[:1]
                            place_boxes_ = place_boxes_[:1]

                        # make the num of pick and place boxes same
                        pick_boxes_, place_boxes_ = self._make_compatible(pick_boxes_, place_boxes_)

                        concepts_ += [concepts[i]] * len(pick_boxes_)
                        if pick not in done_idx:
                            boxes.append(pick_boxes_)
                            picks_ebm_ = [[p_] for p_ in range(count, count + len(pick_boxes_))]
                            picks_ += picks_ebm_
                            count += len(pick_boxes_)
                            done_idx.append(pick)
                            done_ebm.append(picks_ebm_)
                        else:
                            pick_done_idx = done_idx.index(pick)
                            picks_ += done_ebm[pick_done_idx]

                        if place not in done_idx:
                            boxes.append(place_boxes_)
                            place_ebm_ = [[p_] for p_ in range(count, count + len(place_boxes_))]
                            places_ += place_ebm_
                            count += len(place_boxes_)
                            done_idx.append(place)
                            done_ebm.append(place_ebm_)
                        else:
                            place_done_idx = done_idx.index(place)
                            places_ += done_ebm[place_done_idx]
                    boxes = torch.cat(boxes, 0)
                    boxes_ebm = self._pack_boxes_for_ebm(boxes, height, width)

                    assert len(picks_) == len(places_) and len(places_) == len(concepts_)

                    goal_boxes, all_boxes = self._run_ebm(boxes_ebm, picks_, places_, concepts_, move_all=move_all)
                    all_boxes = [self._unpack_boxes_from_ebm(box_.detach().cpu().numpy(), height, width).reshape(-1, 4).numpy() for box_ in all_boxes]
                    video_dict['ebm_boxes'].append(all_boxes)
                    predictions = self._unpack_boxes_from_ebm(goal_boxes, height, width).reshape(-1, 4)
                    predictions = clamp_boxes_torch(predictions, height, width)

                    if move_all:
                        pick_boxes, place_boxes = boxes, predictions
                    else:
                        pick_boxes = torch.cat(
                            [outputs[pick] for pick in np.unique(picks)]
                        )
                        place_boxes = torch.cat(
                            [predictions[p][None] for pick in np.unique(np.array(picks_).reshape(-1, 1), axis=0) for p in pick]
                        )

                    assert len(pick_boxes) == len(place_boxes)

                    outputs.append((pick_boxes, place_boxes))

                    if self.visualize:
                        visualize_outputs.append(
                            self._visualize(
                                img, torch.round(place_boxes.to(torch.float32)),
                                caption=f"put",
                                concept=f"put"
                            )
                        )
                        columns.append(f'put')

                    break

                elif op['op'] == 'multiAryEBM':
                    shape_type, _, _, _ = op['concept']
                    if self.verbose:
                        print(op['op'], shape_type)

                    bboxs = outputs[op['inputs'][0]]
                    height, width = img.shape[:2]
                    boxes_ebm = self._pack_boxes_for_ebm(bboxs, height, width)
                    move_all = True
                    picks_ = [np.arange(len(bboxs))]
                    places_ = [[]]
                    concepts_ = [shape_type]
                    goal_boxes, all_boxes = self._run_ebm(
                        boxes_ebm, picks_, places_, concepts_, move_all=move_all)
                    predictions = self._unpack_boxes_from_ebm(goal_boxes, height, width).reshape(-1, 4)
                    all_boxes = [self._unpack_boxes_from_ebm(box_.detach().cpu().numpy(), height, width).reshape(-1, 4).numpy() for box_ in all_boxes]
                    video_dict['ebm_boxes'].append(all_boxes)

                    video_dict['ebm_captions'].append(f"EBM({shape_type}, {video_dict['captions'][0]})")

                    predictions = clamp_boxes_torch(predictions, height, width)
                    assert len(predictions) == len(bboxs)
                    outputs.append((bboxs, predictions.to(torch.float32)))

                    if self.visualize:
                        visualize_outputs.append(
                            self._visualize(
                                img, predictions,
                                caption=f"{shape_type}",
                                concept=f'{shape_type}'
                            )
                        )
                        columns.append(f'{shape_type}')

                else:
                    assert False, f'unknown op: {op}'

            batch_outputs.append(outputs[-1])

        return batch_outputs, module_outputs, visualize_outputs, columns, video_dict

    @torch.no_grad()
    def _get_programs(self, phrases, use_gt=False,
                      gt_programs=None):
        """Run Seq2Tree Parser, phrases is a list of str."""
        if use_gt:
            return gt_programs
        if not (" and " in phrases[0] and " to the " in phrases[0]):
            _, programs = self.parser(
                phrases, None,
                teacher_forcing=False, compute_loss=False
            )
        else:
            phrases_ = phrases[0].split(" and ")
            _, programs = self.parser(
                phrases_, None,
                teacher_forcing=False, compute_loss=False
            )
            programs = [merge_programs(programs)]

        return programs

    @torch.no_grad()
    def _filter(
        self, image, caption,
        ground_truths=None, use_gt=False
    ):
        return self._filter_pred(image, caption)

    @torch.no_grad()
    def _filter_pred(self, image, caption):
        """Filters objects mentioned in the caption from image

        Args:
            image ([HXWX3]): image in unnormalized pixel format (0-256)
            caption ([string]): language description of object

        Returns:
            Bounding boxes ([NX4]): bboxes satisfying caption (x1, y1, x2, y2)
        """
        # mean-std normalize the input image (batch-size: 1)
        img = (image / 255.).permute(2, 0, 1)
        img = transform(img).unsqueeze(0)

        if 'hole' in caption:
            caption = caption.split("hole")[0].strip() + '_hole'
        caption = f"find {caption}"
        # propagate through the model
        memory_cache = self.bdetr_model(img, [caption], encode_and_save=True)
        outputs = self.bdetr_model(img, [caption], encode_and_save=False, memory_cache=memory_cache)
        # keep only predictions with 0.7+ confidence
        probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
        keep = (probas > 0.7).cpu()

        if keep.sum() == 0:
            keep = (probas == probas.max())

        # sort boxes by confidence
        scores = probas.cpu()[keep]
        boxes = outputs['pred_boxes'].cpu()[0, keep]
        sorted_scores_boxes = sorted(
            zip(scores.tolist(), boxes.tolist()), reverse=True)
        _, sorted_boxes = zip(*sorted_scores_boxes)
        sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])

        #  convert boxes from [0; 1] to image scales
        _, _, h, w = img.shape
        bboxes_scaled = rescale_bboxes(
            sorted_boxes,
            img_h=h, img_w=w)
        return bboxes_scaled

    def _run_ebm(self, boxes, subj, obj, rel, move_all):
        """
        boxes: np.array, (n_relevant_boxes, 4)
        subj: list of lists, rel boxes, e.g. [[1], [0, 9]], points to boxes
        obj: list of lists, ref boxes, e.g. [[2], []], points to boxes
        rel: list of str, relation names, e.g. ["right", "circle"]
        move_all: bool, whether to move all (True) or fix object box (False)

        subj, obj and rel need to have the same number of elements
        if there's no ref object (e.g. in shapes), the corresponding
        element is []

        returns boxes, np.array with same shape as the input boxes
        """
        assert len(subj) == len(obj)
        assert len(obj) == len(rel)
        boxes, all_boxes = langevin(self.ebm_dict, boxes, subj, obj, rel, move_all)
        return boxes.detach().cpu().numpy(), all_boxes

    @staticmethod
    def _make_compatible(pick_boxes_, place_boxes_):
        # make the num of pick and place boxes same
        if len(place_boxes_) < len(pick_boxes_):
            repeat_num = math.ceil(
                (len(pick_boxes_) / float(len(place_boxes_)))
            )
            place_boxes_ = place_boxes_.repeat(repeat_num, 1)

        if len(place_boxes_) > len(pick_boxes_):
            place_boxes_ = place_boxes_[:len(pick_boxes_)]
        return pick_boxes_, place_boxes_

    @staticmethod
    def _crop_img_inside_bbox(img, bbox):
        """
        Inputs:
            img: H, W, 3
            bbox: N, 4 [x1, y1, x2, y2]
        Outputs:
            img_crops: N, h, w, 3
        """
        img_patches = []
        for box in bbox:
            try:
                img_patches.append(img[box[1]: box[3], box[0]: box[2]])
            except Exception as e:
                print(box, e)
        return img_patches

    @staticmethod
    def _visualize(img, detections=None, concept='scene', caption=None):
        all_boxes = []
        if detections is not None:
            for b_i, box in enumerate(detections):
                box_data = {"position": {
                    "minX": box[0].item(),
                    "maxX": box[2].item(),
                    "minY": box[1].item(),
                    "maxY": box[3].item()
                    },
                    "class_id": 0,
                    "domain": "pixel"
                }
                all_boxes.append(box_data)

        box_image = wandb.Image(
            img[..., :3].cpu().numpy(),
            caption=caption,
            boxes={"predictions": {
                "box_data": all_boxes,
                "class_labels": {0: "obj", 1: "rand"}
                }} if detections is not None else None,
            classes=wandb.Classes([{"name": 0, 'id': 'obj'}])
            )
        if detections is None:
            print("bad")
        return box_image

    def _pack_boxes_for_ebm(self, boxes, height, width):
        # boxes are (x1, y1, x2, y2)
        # normalize
        boxes = torch.clone(boxes).to(torch.float32)
        boxes[..., (0, 2)] /= width
        boxes[..., (1, 3)] /= height

        # to center-size
        boxes = torch.stack((
            (boxes[..., 0] + boxes[..., 2]) * 0.5,
            (boxes[..., 1] + boxes[..., 3]) * 0.5,
            boxes[..., 2] - boxes[..., 0],
            boxes[..., 3] - boxes[..., 1]
        ), -1)

        # scale
        boxes[..., 0] = boxes[..., 0]*(XMAX - XMIN) + XMIN
        boxes[..., 1] = boxes[..., 1]*(YMAX - YMIN) + YMIN
        return boxes.cpu().numpy()

    def _unpack_boxes_from_ebm(self, boxes, height, width):
        # boxes are (x, y, W, H) normalized and rescaled
        # boxes = boxes.squeeze(0)  # un-batch
        # un-scale
        boxes[..., 0] = (boxes[..., 0] - XMIN) / (XMAX - XMIN)
        boxes[..., 1] = (boxes[..., 1] - YMIN) / (YMAX - YMIN)

        # to (x1, y1, x2, y2)
        boxes = np.concatenate((
            boxes[..., :2] - boxes[..., 2:] / 2,
            boxes[..., :2] + boxes[..., 2:] / 2
        ), -1)

        # un-normalize
        boxes[..., (0, 2)] *= width
        boxes[..., (1, 3)] *= height
        return torch.from_numpy(boxes)
