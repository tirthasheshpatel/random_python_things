import numpy as np
from manim import *

FRAME_WIDTH = config.frame_width
FRAME_HEIGHT = config.frame_height
PIXEL_WIDTH = config.pixel_width
PIXEL_HEIGHT = config.pixel_height

class GraphSceneWithCamera(GraphScene, MovingCameraScene):
    def setup(self):
        GraphScene.setup(self)
        MovingCameraScene.setup(self)

    def construct(self):
        self.setup_axes(animate=True)
        # save the state of the camera so that
        # you can call `Restore` animation to
        # restore this saved state.
        self.camera_frame.save_state()
        sin_graph = self.get_graph(np.sin, color=RED,
                                   x_min=0, x_max=2*PI)
        tracing_point  = (Dot().move_to(sin_graph.points[0])
                               .set_color(ORANGE))
        starting_point = Dot().move_to(sin_graph.points[0])
        ending_point   = Dot().move_to(sin_graph.points[-1])
        def tracing_func(mobj):
            mobj.move_to(tracing_point.get_center())
        self.play(ShowCreation(sin_graph))
        self.play(ShowCreation(starting_point),
                  ShowCreation(ending_point))
        self.play(self.camera_frame.animate
                  .scale(0.5)
                  .move_to(tracing_point
                           .get_center()))
        self.play(ShowCreation(tracing_point))
        self.camera_frame.add_updater(tracing_func)
        self.play(MoveAlongPath(tracing_point, sin_graph),
                  run_time=2, rate_func=linear)
        self.camera_frame.remove_updater(tracing_func)
        self.play(self.camera_frame.animate.scale(2).move_to(ORIGIN))
        # Alternative way to perform the above animation...
        # self.play(Restore(self.camera_frame))
        all_objs = VGroup(self.axes, sin_graph, tracing_point,
                          starting_point, ending_point)
        self.wait()
        self.play(FadeOut(all_objs))
