"""Sorting Task."""

from tasks.place_red_in_green import PlaceRedInGreen


class PlaceBlueInOrange(PlaceRedInGreen):
    """Sorting Task."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.block_color = 'blue'
        self.bowl_color = 'orange'
        self.lang_template = "put the blue blocks in a orange bowl"
        self.task_completed_desc = "done placing blocks in bowls."
