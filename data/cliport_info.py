ALL_CLASSES = [
    'big brown kit', 'letter E shape_hole', 'letter M shape_hole', 'hexagon_hole', 'letter L shape_hole', 'letter E shape', 'letter M shape', 'hexagon', 'letter L shape', 'ring_hole', 'ring', 'heart_hole', 'heart', 'blue block', 'yellow block', 'blue ring', 'yellow ring', 'gray block', 'brown block', 'cyan cylinder', 'red cylinder', 'red ring', 'cyan ring', 'cyan block', 'yellow cylinder', 'green block', 'blue cylinder', 'red block', 'green ring', 'gray ring', 'gray cylinder', 'brown ring', 'brown cylinder', 'green cylinder', 'orange block', 'orange ring', 'pink block', 'purple block', 'white cylinder', 'white ring', 'white block', 'orange cylinder', 'pink ring', 'pink cylinder', 'purple ring', 'purple cylinder', 'big brown box', 'honey dipper', 'rhino figure', 'porcelain spoon', 'lion figure', 'yoshi figure', 'pepsi max box', 'nintendo cartridge', 'magnifying glass', 'hammer', 'spiderman figure', 'ball puzzle', 'black fedora', 'mario figure', 'scissors', 'nintendo 3ds', 'pepsi next box', 'black shoe with orange stripes', 'tablet', 'spatula with purple head', 'c clamp', 'silver tape', 'grey soccer shoe with cleats', 'butterfinger chocolate', 'alarm clock', 'porcelain salad plate', 'bull figure', 'screwdriver', 'dog statue', 'green and white striped towel', 'black shoe with green stripes', 'red and white flashlight', 'light brown boot with golden laces', 'black sandal', 'orca plush toy', 'office depot box', 'rocket racoon figure', 'can opener', 'android toy', 'red cup', 'black razer mouse', 'toy school bus', 'red and white striped towel', 'white razer mouse', 'porcelain cup', 'dinosaur figure', 'brown fedora', 'frypan', 'crayon box', 'purple tape', 'hard drive', 'pepsi wild cherry box', 'toy train', 'unicorn toy', 'black and blue sneakers', 'pepsi gold caffeine free box', 'black boot with leopard print', 'yellow bowl', 'blue bowl', 'red bowl', 'orange bowl', 'purple bowl', 'green bowl', 'brown bowl', 'pink bowl', 'cyan bowl', 'gray bowl', 'white bowl', 'square_hole',
    'circle_hole',
    'pentagon_hole',
    'flower_hole',
    'square',
    'circle',
    'pentagon',
    'flower',
    'diamond_hole',
    'triangle_hole',
    'rectangle_hole',
    'diamond',
    'triangle',
    'rectangle',
    'letter A shape_hole',
    'letter A shape',
    'star_hole',
    'letter V shape_hole',
    'letter R shape_hole',
    'star',
    'letter V shape',
    'letter R shape',
    'letter T shape_hole',
    'letter T shape',
    'letter G shape_hole',
    'letter G shape',
    'plus_hole',
    'plus'
]

SHAPE = [
    'circle',
    'line',
    'tower',
    'triangle',
    'square'
]

SIZE = [
    'small',
    'medium',
    'large',
    'none'
]

RELATIONS = [
    "inside",
    "left",
    "right",
    "above",
    "below"
]

IS_COMPOSITION = [
    "true",
    "false"
]

# VERTICAL_POS = [
#     'top',
#     'middle',
#     'bottom',
#     'none'
# ]

# HORIZONTAL_POS = [
#     'left',
#     'center',
#     'right',
#     'none'
# ]

POS = [
    'top',
    'center',
    'bottom',
    'left side',
    'right side',
    'top left corner',
    'top right corner',
    'bottom left corner',
    'bottom right corner',
    'none'
]

CORNERS = [
    'front left tip',
    'front right tip',
    'back left corner',
    'back right corner'
]

COMP_ATTRIBUTES = [
    'color',
    'height',
    'material'
]

COMP_VALUES = [
    'smaller',
    'larger',
]

LOCATION = [
    'top',
    'bottom',
    'left',
    'right',
    'middle',
    'none'
]

SAME_GOAL = [
    'align-rope-test',
    'align-rope-train',
    'align-rope-val',
    'packing-boxes-pairs-seen-colors-test',
    'packing-boxes-pairs-seen-colors-train',
    'packing-boxes-pairs-seen-colors-val',
    'packing-boxes-pairs-unseen-colors-test',
    'packing-boxes-pairs-unseen-colors-train',
    'packing-boxes-pairs-unseen-colors-val',
    'packing-seen-google-objects-group-test',
    'packing-seen-google-objects-group-train',
    'packing-seen-google-objects-group-val',
    'packing-unseen-google-objects-group-test',
    'packing-unseen-google-objects-group-train',
    'packing-unseen-google-objects-group-val',
    'put-block-in-bowl-seen-colors-test',
    'put-block-in-bowl-seen-colors-train',
    'put-block-in-bowl-seen-colors-val',
    'separating-piles-seen-colors-test',
    'separating-piles-seen-colors-train',
    'separating-piles-seen-colors-val',
    'separating-piles-unseen-colors-test',
    'separating-piles-unseen-colors-train',
    'separating-piles-unseen-colors-val',
    'circle-seen-colors-val'
]

DIFF_GOAL = [
    'assembling-kits-seq-seen-colors-test',
    'assembling-kits-seq-seen-colors-train',
    'assembling-kits-seq-seen-colors-val',
    'assembling-kits-seq-unseen-colors-test',
    'assembling-kits-seq-unseen-colors-train',
    'assembling-kits-seq-unseen-colors-val',
    'packing-shapes-test',
    'packing-shapes-train',
    'packing-shapes-val',
    'packing-seen-google-objects-seq-test',
    'packing-seen-google-objects-seq-train',
    'packing-seen-google-objects-seq-val',
    'packing-unseen-google-objects-seq-test',
    'packing-unseen-google-objects-seq-train',
    'packing-unseen-google-objects-seq-val',
    'towers-of-hanoi-seq-seen-colors-test',
    'towers-of-hanoi-seq-seen-colors-train',
    'towers-of-hanoi-seq-seen-colors-val',
    'towers-of-hanoi-seq-unseen-colors-test',
    'towers-of-hanoi-seq-unseen-colors-train',
    'towers-of-hanoi-seq-unseen-colors-val',
    'stack-block-pyramid-seq-seen-colors-test',
    'stack-block-pyramid-seq-seen-colors-train',
    'stack-block-pyramid-seq-seen-colors-val',
    'stack-block-pyramid-seq-unseen-colors-test',
    'stack-block-pyramid-seq-unseen-colors-train',
    'stack-block-pyramid-seq-unseen-colors-val',
]


COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'pink', 'white', 'yellow', 'brown', 'gray', 'cyan']
OBJ = ['bowl', 'block', 'ring'] #['cube', 'ring', 'cylinder'] #['bowl', 'block', 'ring']

COLOR_AGNOSTIC_OBJECTS = ['rope', '3-sided frame']

COLOR_OBJECTS = ['blue bowl', 'blue block', 'blue ring', 'red bowl', 'red block', 'red ring', 'green bowl', 'green block', 'green ring', 'orange bowl', 'orange block', 'orange ring', 'purple bowl', 'purple block', 'purple ring', 'pink bowl', 'pink block', 'pink ring', 'white bowl', 'white block', 'white ring', 'yellow bowl', 'yellow block', 'yellow ring', 'brown bowl', 'brown block', 'brown ring', 'gray bowl', 'gray block', 'gray ring', 'cyan bowl', 'cyan block', 'cyan ring', 'brown box']

ONE_COLOR_OBJECTS = ['blue object', 'red object', 'green object', 'orange object', 'purple object', 'pink object', 'white object', 'yellow object', 'brown object', 'gray object', 'cyan object']

GOOGLE_OBJECTS = []
ALL_CLIPORT_OBJECTS = ['blue bowl', 'blue block', 'blue box', 'red bowl', 'red block', 'red box', 'green bowl', 'green block', 'green box', 'orange bowl', 'orange block', 'orange box', 'purple bowl', 'purple block', 'purple box', 'pink bowl', 'pink block', 'pink box', 'white bowl', 'white block', 'white box', 'yellow bowl', 'yellow block', 'yellow box', 'brown bowl', 'brown block', 'brown box', 'gray bowl', 'gray block', 'gray box', 'cyan bowl', 'cyan block', 'cyan box', 'alarm clock', 'android toy', 'ball puzzle', 'black blue sneakers', 'black boot with leopard print', 'black fedora', 'black razer mouse', 'black sandal', 'black shoe with green stripes', 'black shoe with orange stripes', 'brown fedora', 'bull figure', 'butterfinger chocolate', 'c clamp', 'can opener', 'crayon box', 'dinosaur figure', 'dog statue', 'frypan', 'green white striped towel', 'grey soccer shoe with cleats', 'hammer', 'hard drive', 'honey dipper', 'light brown boot with golden laces', 'lion figure', 'magnifying glass', 'mario figure', 'nintendo 3ds', 'nintendo cartridge', 'office depot box', 'orca plush toy', 'pepsi gold caffeine free box', 'pepsi max box', 'pepsi next box', 'pepsi wild cherry box', 'porcelain cup', 'porcelain salad plate', 'porcelain spoon', 'purple tape', 'red white flashlight', 'red white striped towel', 'red cup', 'rhino figure', 'rocket racoon figure', 'scissors', 'screwdriver', 'silver tape', 'spatula with purple head', 'spiderman figure', 'tablet', 'toy school bus', 'toy train', 'unicorn toy', 'white razer mouse', 'yoshi figure', 'alarm clock', 'android toy', 'ball puzzle', 'black blue sneakers', 'black boot with leopard print', 'black fedora', 'black razer mouse', 'black sandal', 'black shoe with green stripes', 'black shoe with orange stripes', 'brown fedora', 'bull figure', 'butterfinger chocolate', 'c clamp', 'can opener', 'crayon box', 'dinosaur figure', 'dog statue', 'frypan', 'green white striped towel', 'grey soccer shoe with cleats', 'hammer', 'hard drive', 'honey dipper', 'light brown boot with golden laces', 'lion figure', 'magnifying glass', 'mario figure', 'nintendo 3ds', 'nintendo cartridge', 'office depot box', 'orca plush toy', 'pepsi gold caffeine free box', 'pepsi max box', 'pepsi next box', 'pepsi wild cherry box', 'porcelain cup', 'porcelain salad plate', 'porcelain spoon', 'purple tape', 'red white flashlight', 'red white striped towel', 'red cup', 'rhino figure', 'rocket racoon figure', 'scissors', 'screwdriver', 'silver tape', 'spatula with purple head', 'spiderman figure', 'tablet', 'toy school bus', 'toy train', 'unicorn toy', 'white razer mouse', 'yoshi figure', 'alarm clock', 'android toy', 'ball puzzle', 'black blue sneakers', 'black boot with leopard print', 'black fedora', 'black razer mouse', 'black sandal', 'black shoe with green stripes', 'black shoe with orange stripes', 'brown fedora', 'bull figure', 'butterfinger chocolate', 'c clamp', 'can opener', 'crayon box', 'dinosaur figure', 'dog statue', 'frypan', 'green white striped towel', 'grey soccer shoe with cleats', 'hammer', 'hard drive', 'honey dipper', 'light brown boot with golden laces', 'lion figure', 'magnifying glass', 'mario figure', 'nintendo 3ds', 'nintendo cartridge', 'office depot box', 'orca plush toy', 'pepsi gold caffeine free box', 'pepsi max box', 'pepsi next box', 'pepsi wild cherry box', 'porcelain cup', 'porcelain salad plate', 'porcelain spoon', 'purple tape', 'red white flashlight', 'red white striped towel', 'red cup', 'rhino figure', 'rocket racoon figure', 'scissors', 'screwdriver', 'silver tape', 'spatula with purple head', 'spiderman figure', 'tablet', 'toy school bus', 'toy train', 'unicorn toy', 'white razer mouse', 'yoshi figure', 'rope', '3-sided frame']