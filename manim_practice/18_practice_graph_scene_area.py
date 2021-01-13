import numpy as np
from manim import *

FRAME_WIDTH = config.frame_width
FRAME_HEIGHT = config.frame_height
PIXEL_WIDTH = config.pixel_width
PIXEL_HEIGHT = config.pixel_height

class PointTracingInGraph(GraphScene):
    def __init__(self, **kwargs):
        GraphScene.__init__(
            self,
            x_min=0,
            x_max=5,
            y_min=0,
            y_max=6,
            x_labeled_nums=[0, 2, 3]
        )

    def construct(self):
        self.setup_axes(animate=True)
        curve1 = self.get_graph(lambda x: 4 * x - x ** 2, color=RED,
                                x_min=0, x_max=4)
        curve2 = self.get_graph(lambda x: 0.8 * x ** 2 - 3 * x + 4,
                                color=GREEN, x_min=0, x_max=4)
        vline1 = self.get_vertical_line_to_graph(2, curve1, DashedLine,
                                                 color=YELLOW)
        vline2 = self.get_vertical_line_to_graph(3, curve1, DashedLine,
                                                 color=YELLOW)
        # `dx_scaling` acts like the step size.
        area1  = self.get_area(curve1, 0, 0.5, dx_scaling=10, area_color=BLUE)
        area2  = self.get_area(curve1, 2, 3, bounded=curve2, area_color=WHITE)

        self.wait()
        self.play(ShowCreation(curve1))
        self.play(ShowCreation(curve2))
        self.play(ShowCreation(area1))
        self.play(ShowCreation(area2))
        self.wait()
