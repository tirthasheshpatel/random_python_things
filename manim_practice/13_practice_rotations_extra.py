from manim import *

FRAME_WIDTH = config.frame_width
FRAME_HEIGHT = config.frame_height
PIXEL_WIDTH = config.pixel_width
PIXEL_HEIGHT = config.pixel_height

class RoatatingLines(Scene):
    def construct(self):
        l1 = Line(LEFT, RIGHT)
        l2 = Line(LEFT, RIGHT)
        l2.set_stroke(PINK, opacity=1)
        ls = VGroup(l1, l2)

        self.wait()
        self.play(ShowCreation(ls))
        self.wait()
        self.play(Rotate(l2,   3 * PI / 4, about_point=RIGHT), run_time=2)
        self.play(Rotate(l2, - 3 * PI / 4, about_point=RIGHT), run_time=2)
        self.wait()
        self.play(ls.animate.scale(0))
        self.wait()
