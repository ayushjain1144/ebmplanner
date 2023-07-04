"""Sorting Task."""

from tasks.place_red_in_green import PlaceRedInGreen


class PlaceCyanInPurple(PlaceRedInGreen):
    """Sorting Task."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.block_color = 'cyan'
        self.bowl_color = 'purple'
        self.lang_template = "put the cyan blocks in a purple bowl"
        self.task_completed_desc = "done placing blocks in bowls."
