import numpy as np
from numpy import array, linspace, arange
from manim import *

FRAME_WIDTH = config.frame_width
FRAME_HEIGHT = config.frame_height
PIXEL_WIDTH = config.pixel_width
PIXEL_HEIGHT = config.pixel_height

class GraphSceneInManim(GraphScene):
    def __init__(self, **kwargs):
        GraphScene.__init__(
            self,
            x_min=-3 * PI,
            x_max=+3 * PI,
            x_axis_width=12,
            x_leftmost_tick=None,
            x_labeled_nums=range(-9, 10, 2),
            x_axis_label="$x$",
            y_min=-1.,
            y_max=+1.,
            y_axis_height=3,
            y_bottom_tick=None,
            y_labeled_nums=array([-1, 1.]),
            y_axis_label="$y$",
            # num_graph_anchor_points=1e8,
            axes_color=GREEN,
            graph_origin=ORIGIN,
            exclude_zero_label=True,
            **kwargs
        )
        self.function_color = RED

    def construct(self):
        self.setup_axes(animate=True)
        sin_graph = self.get_graph(np.sin, self.function_color)
        cos_graph = self.get_graph(np.cos)
        pi_line   = self.get_vertical_line_to_graph(2*PI, cos_graph,
                                                    color=YELLOW)
        sin_graph_label = self.get_graph_label(sin_graph, r"\sin(x)",
                                               x_val=+3*PI, direction=DOWN)
        cos_graph_label = self.get_graph_label(cos_graph, r"\cos(x)",
                                               x_val=-3*PI, direction=DOWN)
        twopi_label = MathTex(r"x = 2\pi")

        label_coord = self.input_to_graph_point(2*PI, cos_graph)
        cos_point = Dot(label_coord)
        twopi_label.next_to(label_coord, UR)

        # The above three lines can also be approximately
        # created by these 2 lines which, at first look,
        # seems more intuitive. But, it will not work as
        # the lines created by the graph may get tansformed
        # later and it may affect the result.

        # cos_point = Dot().align_to(pi_line, UP)
        # twopi_label.next_to(pi_line, [0, pi_line.get_y(), 0] + UR)

        all_objs = VGroup(sin_graph, cos_graph, pi_line, cos_point,
                          twopi_label, sin_graph_label, cos_graph_label,
                          self.axes)

        self.play(ShowCreation(sin_graph))
        self.play(ShowCreation(sin_graph_label))
        self.play(ShowCreation(cos_graph))
        self.play(ShowCreation(cos_graph_label))
        self.play(ShowCreation(pi_line))
        self.play(ShowCreation(cos_point), run_time=0.2)
        self.play(ShowCreation(twopi_label))
        self.wait()
        self.play(all_objs.animate.scale(0))
        self.wait()
