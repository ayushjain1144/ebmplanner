from copy import deepcopy
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import (
    pack_padded_sequence, pad_packed_sequence
)
from transformers import RobertaTokenizer, RobertaModel

from data.cliport_info import (
    SHAPE, SIZE, POS,
    CORNERS, COMP_ATTRIBUTES, COMP_VALUES, LOCATION,
    RELATIONS, IS_COMPOSITION
)


OP_NAME_TO_NUM_OUTPUTS = {
    "detect_objects": 0,
    "filter": 1,
    "relate_compare": 2,
    "align": 2,
    "binaryEBM": 2,
    "multiAryEBM": 1,
    'filter_frame': 1,  # useful for align rope
    "start_composition": 0,
    "end_composition": 0
}
OP_NAME_TO_NUM_CONCEPTS = {
    "detect_objects": 0,
    "filter": 2,
    "relate_compare": 2,
    "align": 2,
    "binaryEBM": 2,
    "multiAryEBM": 4,
    'filter_frame': 0,
    "start_composition": 0,
    "end_composition": 0
}
OP_NAME_TO_NEED_CONCEPT = {
    "detect_objects": False,
    "filter": True,
    "relate_compare": True,
    "align": True,
    "binaryEBM": True,
    "multiAryEBM": True,
    'filter_frame': False,
    "start_composition": False,
    "end_composition": False

}
OP_NAMES = {
    "detect_objects": 0,
    "filter": 1,
    "relate_compare": 2,
    "align": 3,
    "filter_frame": 4,
    "binaryEBM": 5,
    "multiAryEBM": 6,
    "start_composition": 7,
    "end_composition": 8
}
OP_ID_TO_OP_NAME = {op_id: op_name for (op_name, op_id) in OP_NAMES.items()}


def attention(query, key, value):
    "Compute 'Scaled Dot Product Attention'."
    d_k = query.size(-1)
    scores = (
        torch.matmul(query, key.transpose(-2, -1))
        / math.sqrt(d_k)
    )
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


class Seq2TreeTransformer(nn.Module):
    """Utterance to program tree model."""

    def __init__(self):
        """Initialize modules."""
        super().__init__()
        # Encoder-decoder
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.pre_encoder = RobertaModel.from_pretrained('roberta-base').eval()
        for param in self.pre_encoder.parameters():
            param.requires_grad = False
        self.encoder = nn.GRU(768, 256, 1, bidirectional=True)
        self.decoder = Seq2TreeDecoder(2 * 256, len(OP_NAMES.keys()))

    def forward(self, utterances, programs=None,
                teacher_forcing=True, compute_loss=True):
        """
        Forward pass.

        Args:
            utterances (list): list of utterances
            programs (list): list of program annotations (trees)
            teacher_forcing (bool): set to True during training
            compute_loss (bool): whether to compute loss

        Returns:
            loss (tensor): float tensor of loss for this batch
            output (list): program list of operations
        """
        device = next(self.encoder.parameters()).device

        # Pass through encoder
        inputs = self.tokenizer(utterances, return_tensors="pt", padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            lengths = inputs['attention_mask'].sum(1).cpu().int()
            outputs = self.pre_encoder(**inputs)

        # GRU on top of Roberta
        embeddings = pack_padded_sequence(
            outputs.last_hidden_state, lengths,
            batch_first=True, enforce_sorted=False
            )
        gru_outputs_packed, _ = self.encoder(embeddings)
        gru_outputs, lengths = pad_packed_sequence(gru_outputs_packed)
        feat_vecs = gru_outputs[lengths - 1, torch.arange(len(lengths)), :]

        # Tokenize utterance (for copying)
        tokens = [
            self.tokenizer.tokenize(utterance) for utterance in utterances
        ]
        max_length = max(len(t_list) for t_list in tokens)
        tokens = [
            t_list + (['0'] * (max_length - len(t_list))) for t_list in tokens
        ]

        # Call parse on the output of encoder, word embeddings of concepts
        loss, output = self.decoder(
            feat_vecs,
            programs,
            gru_outputs,
            tokens,
            teacher_forcing,
            compute_loss
        )
        output = convert_output_batch_to_seq(output)
        return loss, output


class SpanConceptDecoderAttention(nn.Module):
    """Compute attention scores for concepts."""

    def __init__(self, context_dim, num_ops):
        """Initialize layers."""
        super().__init__()
        self.start_span_layer = nn.Sequential(
            nn.Linear(context_dim*2, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, context_dim)
        )
        self.end_span_layer = deepcopy(self.start_span_layer)
        self.op_embedding_layer = nn.Embedding(num_ops, context_dim)

    def forward(self, context, operations, seqs):
        """
        Compute attention on concepts using the encoded seq.

        Args:
            - context: (B, feat_dim), global feature per sequence
            - operations: (B,), predicted operation classes
            - seqs: (nwords, B, feat_dim), per-word features
        """
        B = context.shape[0]
        op_embedding = self.op_embedding_layer(operations).unsqueeze(1)
        # op_embedding B, 1, 512
        query = torch.cat((context.unsqueeze(1), op_embedding), 1)  # B, 2, 512
        key = seqs.permute(1, 0, 2).clone().detach()  # B, nwords, 512
        value = seqs.permute(1, 0, 2).clone().detach()
        attended_vec, _ = attention(query, key, value)
        attended_vec = attended_vec.reshape(B, -1)  # B, 1024

        start_vecs = self.start_span_layer(attended_vec)
        end_vecs = self.end_span_layer(attended_vec)
        seqs = seqs.permute(1, 0, 2)
        return (
            10 * F.cosine_similarity(seqs, start_vecs.unsqueeze(1), dim=2),
            10 * F.cosine_similarity(seqs, end_vecs.unsqueeze(1), dim=2)
        )


class WordConceptDecoderAttention(nn.Module):
    """Compute attention scores for concepts."""

    def __init__(self, context_dim, num_ops, num_concepts):
        """Initialize layers."""
        super().__init__()
        self.concept_layer = nn.Sequential(
            nn.Linear(context_dim*2, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_concepts)
        )
        self.op_embedding_layer = nn.Embedding(num_ops, context_dim)

    def forward(self, f, op, seqs):
        """Compute attention on concepts using the encoded seq."""
        B = f.shape[0]
        op_embedding = self.op_embedding_layer(op).unsqueeze(1)  # B, 1, 512
        query = torch.cat((f.unsqueeze(1), op_embedding), 1)  # B, 2, 512
        key = seqs.permute(1, 0, 2).clone().detach()  # B, 21, 512
        value = seqs.permute(1, 0, 2).clone().detach()  # B, 21, 512
        attended_vec, _ = attention(query, key, value)  # B, 2, 512
        attended_vec = attended_vec.squeeze(1).reshape(B, -1)  # B, 1024
        return self.concept_layer(attended_vec)


class OPDecoderAttention(nn.Module):
    """
    Input: seqs, f (present state of tree)
    seqs: 21 X B, 512
    f: B, 512
    Output: ops
    """
    def __init__(self, input_dim, num_ops):
        super().__init__()

        # Classify operation
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_ops)
        )

    def forward(self, f, seqs):
        query = f.unsqueeze(1)  # B, 1, 512
        key = seqs.permute(1, 0, 2).clone().detach()  # B, 21, 512
        value = seqs.permute(1, 0, 2).clone().detach()
        attented_seqs, _ = attention(query, key, value)  # B, 1, 512
        attented_seqs = attented_seqs.squeeze(1)

        return self.mlp(attented_seqs)


class ConceptDecoder(nn.Module):
    """Combine span and word decoders in a single class."""

    def __init__(self, input_dim, num_ops):
        super().__init__()
        self.span_concept_decoder = SpanConceptDecoderAttention(
            input_dim, num_ops
        )
        self.shape_concept_decoder = WordConceptDecoderAttention(
            input_dim, num_ops, len(SHAPE)
        )
        self.relation_concept_decoder = WordConceptDecoderAttention(
            input_dim, num_ops, len(RELATIONS)
        )
        self.composition_concept_decoder = WordConceptDecoderAttention(
            input_dim, num_ops, len(IS_COMPOSITION)
        )
        self.size_concept_decoder = WordConceptDecoderAttention(
            input_dim, num_ops, len(SIZE)
        )
        self.pos_concept_decoder = WordConceptDecoderAttention(
            input_dim, num_ops, len(POS)
        )
        self.corner_concept_decoder = WordConceptDecoderAttention(
            input_dim, num_ops, len(CORNERS)
        )
        self.corner_concept_decoder2 = WordConceptDecoderAttention(
            input_dim, num_ops, len(CORNERS)
        )
        self.comp_attr_concept_decoder = WordConceptDecoderAttention(
            input_dim, num_ops, len(COMP_ATTRIBUTES)
        )
        self.comp_value_concept_decoder = WordConceptDecoderAttention(
            input_dim, num_ops, len(COMP_VALUES)
        )
        self.location_concept_decoder = WordConceptDecoderAttention(
            input_dim, num_ops, len(LOCATION)
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, f, y, seqs, tokens, op, loss,
                teacher_forcing=True, compute_loss=True):
        device = next(self.span_concept_decoder.parameters()).device
        starts, ends = self.span_concept_decoder(f, op, seqs)
        sh_words = self.shape_concept_decoder(f, op, seqs)
        rel_words = self.relation_concept_decoder(f, op, seqs)
        comp_words = self.composition_concept_decoder(f, op, seqs)
        sz_words = self.size_concept_decoder(f, op, seqs)
        p_words = self.pos_concept_decoder(f, op, seqs)
        c_words = self.corner_concept_decoder(f, op, seqs)
        c_words2 = self.corner_concept_decoder2(f, op, seqs)
        ca_words = self.comp_attr_concept_decoder(f, op, seqs)
        cv_words = self.comp_value_concept_decoder(f, op, seqs)
        loc_words = self.location_concept_decoder(f, op, seqs)
        if compute_loss:
            target_start_spans = torch.as_tensor([
                program[1][0][1] if program[0] == 'filter' else -1
                for op_, program in zip(op, y)
            ]).to(device)
            target_end_spans = torch.as_tensor([
                program[1][0][2] if program[0] == 'filter' else -1
                for op_, program in zip(op, y)
            ]).to(device)
            target_rels = torch.as_tensor([
                program[1][0]
                if program[0] == 'binaryEBM' and program[1] is not None
                else -1
                for op_, program in zip(op, y)
            ]).to(device)
            target_comps = torch.as_tensor([
                program[1][-1]
                if (program[0] == 'binaryEBM' or program[0] == 'multiAryEBM')
                    and program[1] is not None
                else -1
                for op_, program in zip(op, y)
            ]).to(device)
            target_sh = torch.as_tensor([
                program[1][0]
                if program[0] == 'multiAryEBM' and program[1] is not None
                else -1
                for op_, program in zip(op, y)
            ]).to(device)
            target_sz = torch.as_tensor([
                program[1][1]
                if program[0] == 'multiAryEBM' and program[1] is not None
                else -1
                for op_, program in zip(op, y)
            ]).to(device)
            target_p = torch.as_tensor([
                program[1][2]
                if program[0] == 'multiAryEBM' and program[1] is not None
                else -1
                for op_, program in zip(op, y)
            ]).to(device)
            target_c = torch.as_tensor([
                program[1][0]
                if program[0] == 'align' and program[1] is not None
                else -1
                for op_, program in zip(op, y)
            ]).to(device)
            target_c2 = torch.as_tensor([
                program[1][1]
                if program[0] == 'align' and program[1] is not None
                else -1
                for op_, program in zip(op, y)
            ]).to(device)
            target_ca = torch.as_tensor([
                program[1][1]
                if program[0] == 'relate_compare' and program[1] is not None
                else -1
                for op_, program in zip(op, y)
            ]).to(device)
            target_cv = torch.as_tensor([
                program[1][0]
                if program[0] == 'relate_compare' and program[1] is not None
                else -1
                for op_, program in zip(op, y)
            ]).to(device)
            target_loc = torch.as_tensor([
                program[1][1]
                if program[0] == 'filter' and program[1] is not None
                else -1
                for op_, program in zip(op, y)
            ]).to(device)
            loss += (
                self.loss_fn(starts, target_start_spans).mean()
                + self.loss_fn(ends, target_end_spans).mean()
                + self.loss_fn(rel_words, target_rels).mean()
                + self.loss_fn(comp_words, target_comps).mean()
                + self.loss_fn(sh_words, target_sh).mean()
                + self.loss_fn(sz_words, target_sz).mean()
                + self.loss_fn(p_words, target_p).mean()
                + self.loss_fn(c_words, target_c).mean()
                + self.loss_fn(c_words2, target_c2).mean()
                + self.loss_fn(ca_words, target_ca).mean()
                + self.loss_fn(cv_words, target_cv).mean()
                + self.loss_fn(loc_words, target_loc).mean()
            )

        if teacher_forcing:
            starts = target_start_spans.flatten()
            ends = target_end_spans.flatten()
            rel_words = target_rels.flatten()
            comp_words = target_comps.flatten()
            sh_words = target_sh.flatten()
            sz_words = target_sz.flatten()
            p_words = target_p.flatten()
            c_words = target_c.flatten()
            c_words2 = target_c2.flatten()
            ca_words = target_ca.flatten()
            cv_words = target_cv.flatten()
            loc_words = target_loc.flatten()
        else:
            starts = torch.argmax(starts, dim=1)
            ends = torch.argmax(ends, dim=1)
            rel_words = torch.argmax(rel_words, dim=1)
            comp_words = torch.argmax(comp_words, dim=1)
            sh_words = torch.argmax(sh_words, dim=1)
            sz_words = torch.argmax(sz_words, dim=1)
            p_words = torch.argmax(p_words, dim=1)
            c_words = torch.argmax(c_words, dim=1)
            c_words2 = torch.argmax(c_words2, dim=1)
            ca_words = torch.argmax(ca_words, dim=1)
            cv_words = torch.argmax(cv_words, dim=1)
            loc_words = torch.argmax(loc_words, dim=1)
        concepts = []
        for k in range(len(op)):
            conc_list = []
            if OP_ID_TO_OP_NAME[op[k].item()] == 'filter':
                conc_list.append(''.join(
                    ' ' + token[1:] if ord(token[0]) == 288 else token
                    for token in tokens[k][starts[k].item():1 + ends[k].item()]
                ).strip())
                conc_list.append(LOCATION[int(loc_words[k])])
            elif OP_ID_TO_OP_NAME[op[k].item()] == 'binaryEBM':
                conc_list = [
                    RELATIONS[int(rel_words[k])],
                    IS_COMPOSITION[int(comp_words[k])]
                ]
            elif OP_ID_TO_OP_NAME[op[k].item()] == 'multiAryEBM':
                conc_list = [
                    SHAPE[int(sh_words[k])],
                    SIZE[int(sz_words[k])],
                    POS[int(p_words[k])],
                    IS_COMPOSITION[int(comp_words[k])]
                ]
            elif OP_ID_TO_OP_NAME[op[k].item()] == 'align':
                conc_list = [
                    CORNERS[int(c_words[k])],
                    CORNERS[int(c_words2[k])]
                ]
            elif OP_ID_TO_OP_NAME[op[k].item()] == 'relate_compare':
                conc_list = [
                    COMP_VALUES[int(cv_words[k])],
                    COMP_ATTRIBUTES[int(ca_words[k])]
                ]
            else:
                conc_list = None
            concepts.append(conc_list)
        return loss, concepts


class Seq2TreeDecoder(nn.Module):

    def __init__(self, input_dim, num_ops):
        super().__init__()
        # Constants
        self.input_dim = input_dim
        self.op_id_to_num_outputs = torch.as_tensor(
            [OP_NAME_TO_NUM_OUTPUTS[op_name] for op_name in OP_NAMES]
        )
        self.op_id_to_need_concept = torch.as_tensor(
            [OP_NAME_TO_NEED_CONCEPT[op_name] for op_name in OP_NAMES]
        ).float()
        self.op_decoder = OPDecoderAttention(input_dim, num_ops)

        # Classify concepts
        self.concept_classifier = ConceptDecoder(input_dim, num_ops)

        # Branch encoders
        self.oencoder_0 = nn.GRU(1, input_dim)
        self.oencoder_1 = nn.GRU(1, input_dim)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, f, y, seqs, tokens,
                teacher_forcing=True, compute_loss=True):
        """Call parse."""
        return self.parse(
            f, y, seqs, tokens,
            teacher_forcing, compute_loss
        )

    def parse(self, f, y, seqs, tokens,
              teacher_forcing=True, compute_loss=True):
        """
        Decode embeddings into programs.

        Inputs:
            - f: encoded embeddings (gru_outputs) (b, input_dim),
            - y: NSCL programs as nested sequences
            - seqs
            - tokens: tokennized words of each utterance
            - teacher_forcing,
            - compute_loss
        )
        Return:
            - loss
            - decoded program: list of [op_id, concept_id, branch0, branch1]
        """
        loss = 0.0
        device = next(self.op_decoder.parameters()).device
        # Classify first operation
        op = self.op_decoder(f, seqs)
        if compute_loss:
            target_ops = torch.as_tensor(
                [OP_NAMES[program[0]] for program in y]  # first operation
            ).long().to(device)
            loss += self.loss_fn(op, target_ops)

        if teacher_forcing and compute_loss:
            op = target_ops  # suppose we found it, what's next?
        else:
            op = torch.argmax(op, dim=1)  # we found what we found

        # Classify concepts
        loss, concepts = self.concept_classifier(
            f, y, seqs, tokens, op, loss,
            teacher_forcing, compute_loss
        )
        # Handle branches recursively
        num_branches = self.op_id_to_num_outputs[op]
        branch0_idxs = num_branches > 0
        branch1_idxs = num_branches > 1
        output = [
            [OP_ID_TO_OP_NAME[op[i].item()], concepts[i], None, None]
            for i in range(len(op))
        ]
        # Get encoded features for branch 1
        # output: 1 X B, 512
        f0 = self.oencoder_0(
            op.reshape(1, -1, 1).float(),
            f.reshape(1, -1, self.input_dim)
        )[0].squeeze(0)
        # Get encoded features for branch 2
        # output: 1 X B, 512
        f1 = self.oencoder_1(
            op.reshape(1, -1, 1).float(),
            f.reshape(1, -1, self.input_dim)
        )[0].squeeze(0)
        if teacher_forcing:
            y0 = [y[i][2] for i in range(len(branch0_idxs)) if branch0_idxs[i]]
            y1 = [y[i][3] for i in range(len(branch1_idxs)) if branch1_idxs[i]]
        else:
            # you might think this is a bug, but this just a hack
            # basically, loss in eval can only work
            # once the checkpoint is already good
            # otherwise teacher forcing is necessary
            # if you try fixing it and waste time increment the number
            # of hours wasted below
            # time = 3
            y0 = y
            y1 = y
        tokens0 = [
            tokens[i] for i in range(len(branch0_idxs)) if branch0_idxs[i]
        ]
        tokens1 = [
            tokens[i] for i in range(len(branch1_idxs)) if branch1_idxs[i]
        ]
        # Parse branch 1 if exists
        if torch.any(branch0_idxs):
            loss0, output0 = self.parse(
                f0[branch0_idxs],
                y0,
                seqs[:, branch0_idxs],
                tokens0,
                teacher_forcing,
                compute_loss
            )
        else:
            loss0 = 0
            output0 = None
        # Parse branch 2 if exists
        if torch.any(branch1_idxs):
            loss1, output1 = self.parse(
                f1[branch1_idxs],
                y1,
                seqs[:, branch1_idxs],
                tokens1,
                teacher_forcing,
                compute_loss
            )
        else:
            loss1 = 0
            output1 = None

        if compute_loss:
            loss += loss0 + loss1
        # Store branches' output
        idx = 0
        for i, keep in enumerate(branch0_idxs):
            if keep:
                output[i][2] = output0[idx]
                idx += 1
        idx = 0
        for i, keep in enumerate(branch1_idxs):
            if keep:
                output[i][3] = output1[idx]
                idx += 1

        return loss, output


def convert_output_batch_to_seq(outputs):
    """
    Convert a list of program outputs to program lists.

    Args:
        outputs (list): program trees, predicted by a parser

    Returns:
        program_lists (list): list of program sequences per output
    """
    return [
        _remove_intermediate(_dfs(outputs[i])[0])
        for i in range(len(outputs))
    ]


def _dfs(prog, program_list=None):
    if program_list is None:
        program_list = []
    inputs = []
    if prog[2] is not None:
        _, in0 = _dfs(prog[2], program_list)
        inputs.append(in0)
    if prog[3] is not None:
        _, in1 = _dfs(prog[3], program_list)
        inputs.append(in1)

    idx = len(program_list)
    program_list.append({"op": prog[0], "inputs": inputs})
    if prog[1] is not None:
        program_list[-1]["concept"] = prog[1]
    return program_list, idx


def _remove_intermediate(program_list):
    clear_list = []
    scene_found = False
    num_scenes = -1
    idx_map = np.arange(len(program_list))
    for p, op in enumerate(program_list):
        if op['op'] == 'detect_objects':
            if not scene_found:
                clear_list.append(op)
                scene_found = True
            idx_map[p] = 0
            num_scenes += 1
        else:
            op['inputs'] = [idx_map[inp] for inp in op['inputs']]
            clear_list.append(op)
            idx_map[p] -= num_scenes
    return clear_list
