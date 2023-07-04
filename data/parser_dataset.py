"""Dataset and data loader to train a parser."""

import json
import os
import pickle

import numpy as np
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

from data.cliport_info import (
    SHAPE, SIZE, POS,
    CORNERS, COMP_ATTRIBUTES, COMP_VALUES, LOCATION,
    RELATIONS, IS_COMPOSITION
)
from data.create_cliport_programs import utterance2program_bdetr as utterance2program


class ParserDataset(Dataset):
    """Dataset utilities for parser."""

    def __init__(self, annos_path='data/cliport_program_annos.json',
                 split='train'):
        """Initialize dataset (here for templated utterances)."""
        self.annos_path = annos_path
        self.split = split
        self.annos = self.load_annos()
        self.lang_utils = LangUtils()

    def load_annos(self):
        """Load annotations."""
        with open(self.annos_path) as fid:
            annos = json.load(fid)['annos']
        return [anno for anno in annos if anno['split'] == self.split]

    def load_annos_old(self):
        annos = []
        utterances = []
        for dir_ in os.listdir(self.annos_path):
            # load sentences
            if self.split in dir_ and not (self.split == 'train' and 'unseen' in dir_):
                lang = pickle.load(
                    open(os.path.join(
                        self.annos_path, dir_, 'ebm/lang.pickle'
                ), 'rb'))
                utterances += lang
        annos = [
            {
                'utterance': utt,
                'program': utterance2program(utt)
            } for utt in utterances
            if not utt.lower().startswith(('done', 'solved'))
        ]
        return annos

    def __getitem__(self, index):
        """Get current batch for input index."""
        anno = self.annos[index]
        raw_utterance, program_tree = self.lang_utils.get_program(
            anno['utterance'], anno['program']  # this is a list of sequences
        )
        return {
            "raw_utterance": raw_utterance,
            "program_list": anno['program'],
            "program_tree": program_tree
        }

    def __len__(self):
        """Return number of utterances."""
        return len(self.annos)


def convert_list_to_prog(prog_list):
    """
    Convert a list of operations to a nested sequence.

    Args:
        prog_list (list): program list, each item looks like:
            {'op': 'filter', 'concept': ['table'], 'inputs': [2]}

    Returns:
        program (list): [(op, concept, prog. inp0, prog. inp1)]

    Examples:
        [
            'filter', 'cup', [
                'relate', 'on', [
                    'filter', 'table', [
                        'scene', None, None, None
                    ], None
                ], None
            ], None
        ]
    """
    progs = []
    used = [False] * len(prog_list)
    for p in prog_list:
        key = next((key for key in p if 'concept' in key), None)
        progs.append(
            [
                p["op"],
                p[key] if key is not None else None,
                progs[p["inputs"][0]] if len(p["inputs"]) > 0 else None,
                progs[p["inputs"][1]] if len(p["inputs"]) > 1 else None
            ]
        )
        if len(p["inputs"]) > 0:
            used[p["inputs"][0]] = True
        if len(p["inputs"]) > 1:
            used[p["inputs"][1]] = True

    # Search in reversed order for the op that nests others
    assert (~np.array(used)).sum() == 1
    for i, use in enumerate(reversed(used)):
        if not use:
            return progs[len(used) - i - 1]


class LangUtils:
    """Preprocessing utilities for language."""

    def __init__(self):
        """Initialize vocabulary dictionaries."""
        self.concept2id = {
            'composition': {conc: c for c, conc in enumerate(IS_COMPOSITION)},
            'relation': {conc: c for c, conc in enumerate(RELATIONS)},
            'shape': {conc: c for c, conc in enumerate(SHAPE)},
            'size': {conc: c for c, conc in enumerate(SIZE)},
            'pos': {conc: c for c, conc in enumerate(POS)},
            'corners': {conc: c for c, conc in enumerate(CORNERS)},
            'cattr': {conc: c for c, conc in enumerate(COMP_ATTRIBUTES)},
            'cval': {conc: c for c, conc in enumerate(COMP_VALUES)},
            'loc': {conc: c for c, conc in enumerate(LOCATION)}
        }
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def get_program(self, raw_utterance, raw_program_list):
        """Get program (list and tree) from raw text."""
        raw_utterance = raw_utterance.replace(',', ' ,').replace('.', '')
        raw_utterance = ' '.join(raw_utterance.split())
        bert_concepts = self.concept_span_to_bert(
            raw_utterance, raw_program_list
        )
        raw_program = convert_list_to_prog(raw_program_list)
        raw_program, _ = self.get_program_with_spans(
            raw_program, bert_concepts, used=None
        )
        return raw_utterance, raw_program

    def get_inds_for_bert(self, text):
        """
        Create a map between sentence's tokens and bert tokens.

        Args:
            text (str): raw text
        Returns:
            token2span (list): [range(concept in tokenized space)]
        """
        # Split commas as separate words
        text = text.replace(',', ' ,').replace('.', '')
        text = ' '.join(text.split())
        # Loop upon tokens, add start offset
        token2span = []
        for t, token in enumerate(text.split()):
            offset = 0
            if token2span:
                offset += token2span[-1][-1] + 1
            if t >= 1:
                new_tokens = self.tokenizer.tokenize(' ' + token)
            else:
                new_tokens = self.tokenizer.tokenize(token)
            token2span.append((np.arange(len(new_tokens)) + offset).tolist())
        return token2span

    def search_for_concept_spans(self, utterance, program_list):
        """Identify concept spans in the utterance."""
        # Collect a list of all concepts
        concepts = []
        for op in program_list:
            if op['op'] == 'filter' and 'concept' in op:
                concepts.append(op['concept'][0])
        concepts = sorted(
            [conc.split() for conc in concepts],
            key=lambda t: len(t), reverse=True
        )
        # Detect them in the sentence
        spans = []
        utterance = utterance.split()
        tag_utterance = list(utterance)
        for concept in concepts:
            ans = -1
            for pos in range(len(utterance)):
                if pos + len(concept) > len(utterance):
                    continue
                if not utterance[pos:pos + len(concept)] == concept:
                    continue
                # we've found an instance
                tagged = any(
                    word == 'TAG'
                    for word in tag_utterance[pos:pos + len(concept)]
                )
                if not tagged:  # this is the first encounter
                    ans = (pos, pos + len(concept) - 1)
                    for i in range(pos, pos + len(concept)):
                        tag_utterance[i] = 'TAG'
                    break
                else:  # already found, save ans but don't break
                    ans = (pos, pos + len(concept) - 1)
            spans.append(ans)
        return zip(concepts, spans)

    def concept_span_to_bert(self, utterance, program_list):
        """Translate concepts to bert spans."""
        token2span = self.get_inds_for_bert(utterance)
        concepts_spans = self.search_for_concept_spans(utterance, program_list)
        return [
            (concept, token2span[span[0]][0], token2span[span[1]][-1])
            for (concept, span) in concepts_spans
        ]

    def get_program_with_spans(self, prog, concepts, used=None):
        """Loop over a nested sequence to replace concepts with spans."""
        if prog is None:
            return None, used
        if used is None:
            used = np.zeros(len(concepts))
        if prog[0] == 'filter' and prog[1] is not None:
            span_list = []
            # 1st item is transformed to a span
            item = prog[1][0]
            ind = next((
                c for c, (concept, u) in enumerate(zip(concepts, used))
                if ' '.join(concept[0]) == item and not u
            ), None)
            if ind is None:  # relax the search
                ind = next(
                    c for c, concept in enumerate(concepts)
                    if ' '.join(concept[0]) == item
                )
            span_list.append(concepts[ind])
            used[ind] = 1
            # 2nd is in LOCATION
            span_list.append(self.concept2id['loc'][prog[1][1]])
            prog[1] = span_list

        elif prog[0] == 'multiAryEBM':
            prog[1] = [
                SHAPE.index(prog[1][0]),
                SIZE.index(prog[1][1]),
                POS.index(prog[1][2]),
                IS_COMPOSITION.index(prog[1][3])
            ]
        # be careful, this is hacky
        # this assumes that different concept lists don't share any value
        elif prog[1] is not None:
            conc_list = []
            for item in prog[1]:
                key = next(
                    key for key, val in self.concept2id.items() if item in val
                )
                conc_list.append(self.concept2id[key][item])
            prog[1] = list(conc_list)
        prog[2], used = self.get_program_with_spans(prog[2], concepts, used)
        prog[3], used = self.get_program_with_spans(prog[3], concepts, used)
        return prog, used


def parser_collate_fn(batch):
    """
    Collate function for seq2tree parser.

    Returns:
        batch (dict): like:
            {
                "raw_utterances": raw utterance strings,
                "program_lists": list of dicts, seq of operations,
                "program_trees": programs as nested operations
            }

    Examples:
        - raw_utterance: 'find the lamp that is near the tv'
        - program_lists: [
            {'inputs': [], 'op': 'scene'},
            {'concept': ['tv'], 'inputs': [0], 'op': 'filter'},
            {'inputs': [1], 'op': 'relate', 'relational_concept': ['near']},
            {'concept': ['lamp'], 'inputs': [2], 'op': 'filter'}
        ]
        - program_trees: [
            'filter', 0, [
                'relate', 2, ['filter', 1, [
                    'scene', None, None, None
                ], None], None
            ], None]
    """
    return {
        "raw_utterances": [ex["raw_utterance"] for ex in batch],
        "program_lists": [ex["program_list"] for ex in batch],
        "program_trees": [ex["program_tree"] for ex in batch]
    }


if __name__ == "__main__":
    PD = ParserDataset(split='train')
    # sentence = 'put the toy school bus inside the magenta , glass box and place it in the center of the circle made of objects that are larger than the yellow block'
    # program_list = [{'op': 'detect_objects', 'inputs': []}, {'op': 'filter', 'concept': ['toy school bus', 'none'], 'inputs': [0]}, {'op': 'filter', 'concept': ['magenta , glass box', 'none'], 'inputs': [0]}, {'op': 'filter', 'concept': ['yellow block', 'none'], 'inputs': [0]}, {'op': 'relate_compare', 'concept': ['larger', 'height'], 'inputs': [0, 3]}, {'op': 'binaryEBM', 'concept': ['inside', 'true'], 'inputs': [1, 2]}, {'op': 'multiAryEBM', 'concept': ['circle', 'none', 'none', 'true'], 'inputs': [4]}, {'op': 'binaryEBM', 'concept': ['inside', 'true'], 'inputs': [5, 6]}]

    # # st()
    # raw_utt, program_tree = PD.lang_utils.get_program(
    #     sentence, program_list
    # )
    # print(program_tree)
    for i in range(len(PD.annos)):
        data = PD.__getitem__(i)