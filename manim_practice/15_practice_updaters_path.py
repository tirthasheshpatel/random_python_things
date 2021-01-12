from manim import *

FRAME_WIDTH = config.frame_width
FRAME_HEIGHT = config.frame_height
PIXEL_WIDTH = config.pixel_width
PIXEL_HEIGHT = config.pixel_height

class TracingPath(Scene):
    def construct(self):
        path = VMobject()
        point = Dot().shift(LEFT)
        path.set_points_as_corners(2 * [point.get_center()])

        def _tracing_func(_path: VMobject) -> None:
            _path.add_points_as_corners([point.get_center()])

        self.wait()
        self.add(path)
        self.play(ShowCreation(point))
        path.add_updater(_tracing_func)
        self.play(Rotate(point, PI, about_point=ORIGIN))
        self.play(point.animate.shift(2 * UP))
        self.play(point.animate.shift(2 * LEFT))
        self.play(point.animate.shift(2 * DOWN))
        path.remove_updater(_tracing_func)
        self.play(FadeOut(point))
        self.play(path.animate.set_fill(PINK, opacity=1))
        self.wait()
        self.play(path.animate.scale(0))
        self.wait()
