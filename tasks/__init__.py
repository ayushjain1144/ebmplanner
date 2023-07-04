"""Ravens tasks."""

from tasks.assembling_kits import AssemblingKits
from tasks.assembling_kits import AssemblingKitsEasy
from tasks.assembling_kits_seq import AssemblingKitsSeqSeenColors
from tasks.assembling_kits_seq import AssemblingKitsSeqUnseenColors
from tasks.assembling_kits_seq import AssemblingKitsSeqFull
from tasks.block_insertion import BlockInsertion
from tasks.block_insertion import BlockInsertionEasy
from tasks.block_insertion import BlockInsertionNoFixture
from tasks.block_insertion import BlockInsertionSixDof
from tasks.block_insertion import BlockInsertionTranslation
from tasks.packing_boxes import PackingBoxes
from tasks.packing_shapes import PackingShapes
from tasks.packing_boxes_pairs import PackingBoxesPairsSeenColors
from tasks.packing_boxes_pairs import PackingBoxesPairsUnseenColors
from tasks.packing_boxes_pairs import PackingBoxesPairsFull
from tasks.packing_google_objects import PackingSeenGoogleObjectsSeq
from tasks.packing_google_objects import PackingUnseenGoogleObjectsSeq
from tasks.packing_google_objects import PackingSeenGoogleObjectsGroup
from tasks.packing_google_objects import PackingUnseenGoogleObjectsGroup
from tasks.place_red_in_green import PlaceRedInGreen
from tasks.place_blue_in_orange import PlaceBlueInOrange
from tasks.place_cyan_in_purple import PlaceCyanInPurple
from tasks.put_block_in_bowl import PutBlockInBowlSeenColors
from tasks.put_block_in_bowl import PutBlockInBowlUnseenColors
from tasks.put_block_in_bowl import PutBlockInBowlFull
# from tasks.task import Task
from tasks.shape import (
    CircleSeenColors, CircleUnseenColors,
    LineSeenColors, LineUnseenColors, SquareSeenColors, SquareUnseenColors, TriangleSeenColors, TriangleUnseenColors
)
from tasks.spatial_relations import (
    AboveSeenColors, AboveUnseenColors, BelowSeenColors, \
    BelowUnseenColors, LeftSeenColors, LeftUnseenColors, \
    RightSeenColors, RightUnseenColors
)
from tasks.compositional_relations import (
    CompositionalRelationsSeenColors, CompositionalRelationsUnSeenColors
)

from tasks.multi_compose_relations import (
    MultiCompositionalRelationsSeenColors, MultiCompositionalRelationsUnseenColors
)

names = {
    # demo conditioned
    'assembling-kits': AssemblingKits,
    'assembling-kits-easy': AssemblingKitsEasy,
    'block-insertion': BlockInsertion,
    'block-insertion-easy': BlockInsertionEasy,
    'block-insertion-nofixture': BlockInsertionNoFixture,
    'block-insertion-sixdof': BlockInsertionSixDof,
    'block-insertion-translation': BlockInsertionTranslation,
    'packing-boxes': PackingBoxes,
    'place-red-in-green': PlaceRedInGreen,
    'place-blue-in-orange': PlaceBlueInOrange,
    'place-cyan-in-purple': PlaceCyanInPurple,

    # shapes
    'circle-seen-colors': CircleSeenColors,
    'circle-unseen-colors': CircleUnseenColors,
    'line-seen-colors': LineSeenColors,
    'line-unseen-colors': LineUnseenColors,
    'triangle-seen-colors': TriangleSeenColors,
    'triangle-unseen-colors': TriangleUnseenColors,
    'square-seen-colors': SquareSeenColors,
    'square-unseen-colors': SquareUnseenColors,
    'triangle-seen-colors': TriangleSeenColors,
    'triangle-unseen-colors': TriangleUnseenColors,

    # relations
    'left-seen-colors': LeftSeenColors,
    'left-unseen-colors': LeftUnseenColors,
    'right-seen-colors': RightSeenColors,
    'right-unseen-colors': RightUnseenColors,
    'above-seen-colors': AboveSeenColors,
    'above-unseen-colors': AboveUnseenColors,
    'below-seen-colors': BelowSeenColors,
    'below-unseen-colors': BelowUnseenColors,

    # Composition of relations
    'composition-seen-colors': CompositionalRelationsSeenColors,
    'composition-unseen-colors': CompositionalRelationsUnSeenColors,

    # Multi Compose relations
    'composition-seen-colors-group': MultiCompositionalRelationsSeenColors,
    'composition-unseen-colors-group': MultiCompositionalRelationsUnseenColors,

    # goal conditioned
    'assembling-kits-seq-seen-colors': AssemblingKitsSeqSeenColors,
    'assembling-kits-seq-unseen-colors': AssemblingKitsSeqUnseenColors,
    'assembling-kits-seq-full': AssemblingKitsSeqFull,
    'packing-shapes': PackingShapes,
    'packing-boxes-pairs-seen-colors': PackingBoxesPairsSeenColors,
    'packing-boxes-pairs-unseen-colors': PackingBoxesPairsUnseenColors,
    'packing-boxes-pairs-full': PackingBoxesPairsFull,
    'packing-seen-google-objects-seq': PackingSeenGoogleObjectsSeq,
    'packing-unseen-google-objects-seq': PackingUnseenGoogleObjectsSeq,
    'packing-seen-google-objects-group': PackingSeenGoogleObjectsGroup,
    'packing-unseen-google-objects-group': PackingUnseenGoogleObjectsGroup,
    'put-block-in-bowl-seen-colors': PutBlockInBowlSeenColors,
    'put-block-in-bowl-unseen-colors': PutBlockInBowlUnseenColors,
    'put-block-in-bowl-full': PutBlockInBowlFull,
}
