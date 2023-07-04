"""Create compositional dataset using a grammar."""

import argparse
import json
import random
import re
import pickle
from copy import copy
import numpy as np

from data.cliport_info import ALL_CLIPORT_OBJECTS


GRAMMAR = """
S -> REARRANGE | PUT
REARRANGE -> 'rearrange' TARGET_OBJECTS 'into' 'a' SIZE 'sized' SHAPE PLACE_LOCATION
TARGET_OBJECTS -> 'OBJECT' | RELATE_PROPERTY 'objects' | 'objects' 'that 'are' RELATE 'than' 'the' RELATE_PROPERTY PROPERTY_OBJECTS 
RELATE_PROPERTY -> COLOR | MATERIAL | COLOR ',' MATERIAL
MATERIAL -> 'glass' | 'metal' | 'plastic' 
COLOR -> 'blue' | 'cyan' | 'green' | 'magenta' | 'red' | 'yellow' 
RELATE -> 'smaller' | 'larger'
SIZE -> 'small' | 'medium' | 'large'
SHAPE -> 'circle' | 'line' | 'tower' | 'square' | 'triangle'
POS -> 'top' | 'center' | 'bottom' | 'left' 'side' | 'right' 'side' | 'top' 'left' 'corner' | 'top' 'right' 'corner' | 'bottom' 'left' 'corner' | 'bottom' 'right' 'corner'
PLACE_LOCATION -> 'at' 'the' POS 'of' 'the' 'table' | REL_PHR
REL_PHR -> REL1 'the' REL_OBJ | 'inside' 'the' RELATE_PROPERTY INSIDE_OBJECT CONJ
CONJ -> 'and' 'place' 'it' REL_PHR2 | 
REL_PHR2 -> REL1 'the' REL_OBJ | 'in' 'the' 'center' 'of' 'the' SHAPE 'made' 'of' TARGET_OBJECTS
REL1 -> 'on' 'the 'left 'of' | 'on' 'the' 'right' 'of' | 'above' | 'below'
INSIDE_OBJECT -> 'plate' | 'box'
PUT -> 'put' 'the' TARGET_OBJECTS REL_PHR
PROPERTY_OBJECTS -> 'bowl' | 'block' | 'ring' | 'cube' | 'cylinder'
REL_OBJ -> RELATE_PROPERTY PROPERTY_OBJECTS | 'OBJECT'
"""

GRAMMAR = """
S ->  PUT
COLOR -> 'blue' | 'cyan' | 'green' | 'magenta' | 'red' | 'yellow' 
SHAPE -> 'circle' | 'line' | 'tower' | 'square' | 'triangle'
REL_PHR -> 'to' 'the' REL1 'of' 'the' REL_OBJ 
REL1 -> 'left' | 'right' | 'above' | 'below'
PUT -> 'put' 'the' REL_OBJ REL_PHR
PROPERTY_OBJECTS -> 'bowl' | 'block' | 'ring' | 'cube' | 'cylinder'
REL_OBJ -> COLOR PROPERTY_OBJECTS | 'OBJECT'
"""


def infer_rule(rule, rules):
    """
    Infer a grammar rule, using the other rules if needed.

    Args:
        rule (str): name of the rule, e.g. 'REL_PHR'.
        rules (dict): {rule_name: list of different ways of inference}.

    Returns:
        utterance (str): an utterance containing tags

    Examples:
        To infer REL_PHR -> R1 | R2 | R3, call
        infer_rule('REL_PHR', rules)
        where R1, R2, R3 in rules.
        Also, for a rule REL_PHR -> R1 | R2 | R3,
        rules[REL_PHR] = [R1, R2, R3].
    """
    rand = random.random()
    num_branches = len(rules[rule])
    for i in range(num_branches):
        if rand >= i / num_branches and rand < (i + 1) / num_branches:
            branch = i
    utterance = []
    for word in rules[rule][branch].split():
        if word.startswith("'"):
            utterance.append(word.strip("'"))
        else:  # rule
            utterance.append(infer_rule(word, rules))
    return ' '.join(utterance)


def grammar_infer(grammar, split):
    """Run a probabilistic grammar to produce an utterance."""
    grammar = grammar.strip('\n').split('\n')
    sent_rule = grammar[0].split(' -> ')[0]
    rules = {
        line.split(' -> ')[0]: line.split(' -> ')[1].split(' | ')
        for line in grammar
    }
    utterance = infer_rule(sent_rule, rules)
    repl_utterance = []
    WORD_LIST = copy(ALL_CLIPORT_OBJECTS)
    for word in utterance.split():
        if word == 'OBJECT':
            sample_object = random.choice(WORD_LIST)
            WORD_LIST.remove(sample_object)
            repl_utterance.append(sample_object)
        else:
            repl_utterance.append(word)
    return ' '.join(repl_utterance).strip()


class Node:
    """Node class to contruct program trees."""

    def __init__(self, name, depth):
        """Initialize with node name and program depth at this level."""
        self.name = name
        self.depth = depth
        self.next0 = None
        self.next1 = None
        self.prev = None
        self.rel = None


def complete_tree(subtree, utt_list):
    """
    Build a tree under a given node.

    Args:
        subtree (Node): current node
        utt_list (list): list of utterances to examine

    Returns:
        subtree (Node): built tree
        utt_list (list): list of utterances not examined yet
    """
    if not utt_list:
        return subtree, utt_list
    utterance = utt_list[0]
    if utterance.startswith('and'):
        return subtree, utt_list
    if utterance.split()[0] in {'which', 'that'}:
        rel = utterance.split()[2]
        if not rel.startswith('BETWEEN'):
            child = Node(utterance.split()[-1], subtree.depth + 1)
            child.prev = subtree
            subtree.rel = rel
            subtree.next0, utt_list = complete_tree(child, utt_list[1:])
        elif 'and' not in utterance.split():
            child = Node(utterance.split()[4], subtree.depth + 1)
            child.prev = subtree
            subtree.rel = rel
            subtree.next0, utt_list = complete_tree(child, utt_list[1:])
            child = Node(utt_list[0].split()[-1], subtree.depth + 1)
            child.prev = subtree
            subtree.next1, utt_list = complete_tree(child, utt_list[1:])
        else:  # between obj1 and obj2 (should never reach this point)
            subtree.rel = rel
            subtree.next0 = Node(utterance.split()[4], subtree.depth + 1)
            subtree.next0.prev = subtree
            child = Node(utterance.split()[-1], subtree.depth + 1)
            child.prev = subtree
            subtree.next1, utt_list = complete_tree(child, utt_list[1:])
    elif utterance.split()[0] == 'with':
        child = Node(
            next(word for word in utterance.split() if word.startswith('OBJ')),
            subtree.depth + 1
        )
        child.prev = subtree
        subtree.rel = (
            'INV_'
            + next(word for word in utterance.split() if 'REL' in word)
        )
        subtree.next0 = child
        utt_list = utt_list[1:]
    return subtree, utt_list


def parse_inferred(utterance):
    """Get a program tree for a given utterance."""

    # Split utterance based on 'interesting' words
    wois = ['which', 'that', 'and', 'with']
    split_utt = utterance.split()
    breakpoints = [0] + [
        w for w, word in enumerate(split_utt) if word in wois
    ] + [len(split_utt)]
    utterance_list = [
        ' '.join(split_utt[breakpoints[i]:breakpoints[i + 1]]).strip(' , ')
        for i in range(len(breakpoints) - 1)
    ]
    tree = Node(utterance_list[0].split()[-1], 0)
    tree, _ = complete_tree(tree, utterance_list[1:])
    return tree


def nodes_from_tree(subtree):
    """Convert tree to list of successive nodes."""
    nodes = [subtree]
    if subtree.next0 is not None:
        nodes += nodes_from_tree(subtree.next0)
    if subtree.next1 is not None:
        nodes += nodes_from_tree(subtree.next1)
    return nodes


def utterance2program_bdetr(utterance, cliport=True):
    program = [
        {'op': 'detect_objects', 'inputs': []}
    ]

    if utterance == "rearrange all small objects into a circle inside the plate":
        program = utterance2program_bdetr('rearrange all small objects into a circle')
        program.insert(2, {"op": "filter", "concept": ['plate', 'none'], "inputs": [0]})
        return program

    if utterance.startswith('align'):
        program += [
            {"op": "filter", "concept": [utterance.split()[2], 'none'], "inputs": [0]},
            {"op": "filter_frame", "inputs": [0]},
        ]
        directions = utterance.split('from')[1].split('to')
        program.append(
            {'op': 'align', 'concept': [directions[0].strip(), directions[1].strip()], 'inputs': [1, 2]}
        )
    elif utterance.startswith('put') and cliport:
        objects = re.split(' in | on ', utterance)
        objects[1] = ' '.join(objects[1].split()[1:])
        loc = "none"
        obj2 = objects[1]

        program += [
            {"op": "filter", "concept": [' '.join(objects[0].split()[2:]), 'none'], "inputs": [0]},
            {"op": "filter", "concept": [obj2.strip(), loc], "inputs": [0]},
            {'op': 'binaryEBM', "concept": ["inside", "false"], 'inputs': [1, 2]}
        ]
    elif ' and ' in utterance and ' to the ' in utterance:
        subsubutt = utterance.split(' and ')
        subsubutt = [s.strip() for s in subsubutt]

        # extract objs and relations
        programs = []
        for utt_ in subsubutt:
            program_ = utterance2program_bdetr(utt_, cliport=False)
            programs.append(program_)

        return merge_programs(programs)

    elif utterance.startswith('put') and not cliport:
        subutt = re.split(' inside | on | above | below | to', utterance)
        subutt[0] = ' '.join(subutt[0].split()[2:])
        if 'that are' in subutt[0]:
            relate_utt = subutt[0].split('than')
            relation = relate_utt[0].split()[-1]
            obj2 = ' '.join(relate_utt[1].split()[1:])
            program += [
                {"op": "filter", "concept": [obj2.strip(), 'none'], "inputs": [0]},
                {"op": 'relate_compare', "concept": [relation.strip(), "height"], "inputs": [0, 1]}
            ]
            depth = 2
        else:
            program += [
                {"op": "filter", "concept": [subutt[0].strip(), 'none'], "inputs": [0]},
            ]
            depth = 1
        rel_phr = utterance.split(subutt[0].strip())[1].strip()
        if rel_phr.startswith(('on', 'above', 'below', 'to')) and 'and' not in rel_phr:
            rel_obj = rel_phr.split('the')[-1]
            rel = ' '.join(rel_phr.split('the')[:-1])
            if 'left' in rel:
                relation = 'left'
            elif 'right' in rel:
                relation = 'right'
            elif 'above' in rel:
                relation  = 'above'
            elif 'below' in rel:
                relation = 'below'
            else:
                assert False, rel
            program += [
                {"op": "filter", "concept": [rel_obj.strip(), 'none'], "inputs": [0]},
                {"op": "binaryEBM", "concept": [relation, "false"], "inputs": [depth, depth+1]},
            ]
            depth += 2

        else:
            obj = rel_phr.split('the')[-1]
            program += [
                {"op": "filter", "concept": [obj.strip(), 'none'], "inputs": [0]},
            ]
            depth += 1
            program += [
                {"op": "binaryEBM", "concept": ["inside", "false"], "inputs": [depth-1, depth]},
            ]
            depth += 1

    elif utterance.startswith('pack'):
        objects = re.split(' inside | into | in ', utterance)
        if objects[0].startswith('pack all'):
            obj0 = ' '.join(objects[0].split()[1:])
        else:
            obj0 = ' '.join(objects[0].split()[2:])
        obj1 = ' '.join(objects[1].split()[1:])
        program += [
            {"op": "filter", "concept": [obj0.strip(), 'none'], "inputs": [0]},
            {"op": "filter", "concept": [obj1.strip(), 'none'], "inputs": [0]},
            {'op': 'binaryEBM', "concept": ["inside", "false"], 'inputs': [1, 2]}
        ]
    elif utterance.startswith('push'):
        objects = utterance.split('into')
        obj0 = ' '.join(objects[0].split()[2:])
        obj1 = ' '.join(objects[1].split()[1:])
        program += [
            {"op": "filter", "concept": [obj0.strip(), 'none'], "inputs": [0]},
            {"op": "filter", "concept": [obj1.strip(), 'none'], "inputs": [0]},
            {'op': 'binaryEBM', "concept": ["inside", "false"], 'inputs': [1, 2]}
        ]
    elif utterance.startswith('move'):
        objects = utterance.split('to')
        obj0 = ' '.join(objects[0].split()[2:])
        obj1 = ' '.join(objects[1].split()[1:])
        program += [
            {"op": "filter", "concept": [obj0.strip(), 'none'], "inputs": [0]},
            {"op": "filter", "concept": [obj1.strip(), 'none'], "inputs": [0]},
            {'op': 'binaryEBM', "concept": ["inside", "false"], 'inputs': [1, 2]}
        ]

    elif utterance.startswith('rearrange'):
        subutt = re.split(' into | circle | line | tower | square | triangle | rearrange', utterance)
        obj = subutt[0].split("rearrange")[-1].strip()
        program += [
            {"op": "filter", "concept": [obj, 'none'], "inputs": [0]},
        ]
        depth = 1
        shape = subutt[1].split('a')[-1].strip()
        program += [
            {"op": "multiAryEBM", "concept": [shape.strip(), 'none', 'none', "false"], "inputs": [depth]}
        ]

    else:
        assert False, utterance
    return program


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


def merge_programs(programs):
    merged_program = []
    merged_program.append(programs[0][0])

    filters = []
    actions = []
    for program in programs:
        for prog in program:
            if prog['op'] == 'detect_objects':
                continue
            elif prog['op'] == 'filter':
                if prog not in filters:
                    filters.append(prog)
            else:
                try:
                    pick, place = prog['inputs']
                except Exception as e:
                    from ipdb import set_trace as st
                    st()
                pick_filter = program[pick]
                pick_index = filters.index(pick_filter) + 1

                place_filter = program[place]
                place_index = filters.index(place_filter) + 1

                prog['inputs'] = [pick_index, place_index]
                actions.append(prog)

    merged_program.extend(filters)
    merged_program.extend(actions)

    return merged_program


def convert_prog_to_list(prog):
    """
    Convert a list of program outputs to program lists.

    Args:
        outputs (list): program trees, predicted by a parser

    Returns:
        program_lists (list): list of program sequences per output
    """
    return _remove_intermediate(_dfs(prog)[0])


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
            op['inputs'] = [int(idx_map[inp]) for inp in op['inputs']]
            clear_list.append(op)
            idx_map[p] -= num_scenes
    return clear_list


def _convert_output_to_list(output, tag_dict):
    if not output:
        return None
    return [
        output[0],
        [tag_dict[output[1]]] if output[1] is not None else None,
        _convert_output_to_list(output[2], tag_dict),
        _convert_output_to_list(output[3], tag_dict)
    ]


if __name__ == "__main__":
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--is_relations", action='store_true')
    argparser.add_argument("--is_shape", action='store_true')
    argparser.add_argument("--ndemos", default=100, type=int)

    args = argparser.parse_args()

    # Collect utterances and programs for structformer
    annos = []

    # # Collect data for cliport
    home_dir = 'benchmark_data'
    if args.is_relations:
        dirs = [
            'left-seen-colors',
            'right-seen-colors',
            'above-seen-colors',
            'below-seen-colors',
            'left-unseen-colors',
            'right-unseen-colors',
            'above-unseen-colors',
            'below-unseen-colors'
        ]
    elif args.is_shape:
        dirs = [
            'circle-seen-colors',
            'circle-unseen-colors',
            'line-seen-colors',
            'line-unseen-colors'
        ]
    else:
        dirs = [
            'assembling-kits-seq-seen-colors',
            'packing-seen-google-objects-group',
            'packing-seen-google-objects-seq',
            'put-block-in-bowl-seen-colors',
            'assembling-kits-seq-unseen-colors',
            'packing-unseen-google-objects-group',
            'packing-unseen-google-objects-seq',
            'put-block-in-bowl-unseen-colors'
        ]

    if args.is_relations:
        dtype = "relations"
    elif args.is_shape:
        dtype = "shapes"
    else:
        dtype = "cliport"
    for split in ["train", "val", 'test']:
        for dir in dirs:
            path = f"{home_dir}/{dir}-{split}/ebm/lang_{dtype}_{args.ndemos}.pickle"
            fp = pickle.load(open(path, 'rb'))
            for sent in fp:
                if sent.startswith("done"):
                    continue
                program = utterance2program_bdetr(sent, cliport=not (args.is_relations or args.is_shape))
                annos.append({
                    'utterance': sent,
                    'program': program,
                    'split': split
                })

    random.shuffle(annos)

    with open(f'cliport_program_annos_{dtype}_{args.ndemos}.json', 'w') as fid:
        json.dump({
            'annos': annos,
        }, fid, indent=4)
