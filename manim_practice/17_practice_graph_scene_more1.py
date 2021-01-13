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
            x_min=-10,
            x_max=10,
            y_min=-1,
            y_max=1,
            x_axis_width=FRAME_WIDTH,
            y_axis_height=3,
            graph_origin=ORIGIN,
            axes_color=GREEN,
        )

    def construct(self):
        self.setup_axes(animate=True)

        sin_graph = self.get_graph(np.sin, color=RED, x_min=-10, x_max=10)

        point_init_coords = self.input_to_graph_point(-10, sin_graph)
        point = Dot(point_init_coords).scale(0.1)
        _p_mat = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
        following_arrow = Arrow(_p_mat @ point.get_center(),
                                point.get_center(), buff=0)\
                                .set_color(ORANGE)

        all_objs = VGroup(sin_graph, point, self.axes)

        def _tracing_func(_arrow: Arrow) -> None:
            _arrow.become(Arrow(_p_mat @ point.get_center(),
                                point.get_center(), buff=0)
                                .set_color(ORANGE))
            # BELOW: alternative way to move the arrow ahead...
            # This is not preferred as it doesn't update the size of
            # the arrow head!!! DON'T USE IT!!!!
            # _arrow.put_start_and_end_on(_p_mat @ point.get_center(),
            #                             point.get_center())

        self.play(ShowCreation(sin_graph))
        following_arrow.add_updater(_tracing_func)
        self.add(following_arrow)
        self.play(MoveAlongPath(point, sin_graph), run_time=6,
                                rate_func=linear)
        following_arrow.remove_updater(_tracing_func)
        self.remove(following_arrow)
        self.wait()
        self.play(FadeOut(all_objs))
        self.wait()
