from manim import *

FRAME_WIDTH = config.frame_width
FRAME_HEIGHT = config.frame_height
PIXEL_WIDTH = config.pixel_width
PIXEL_HEIGHT = config.pixel_height

class MoreAnimations(Scene):
    def construct(self):
        origin  = Dot(ORIGIN)
        circle1 = Circle()
        circle2 = Circle().shift(2 * RIGHT)
        circles = VGroup(circle1, circle2)

        self.play(GrowFromCenter(circles))
        self.play(origin.animate.shift(RIGHT))
        self.play(MoveAlongPath(origin, circle1), run_time=2,
                  rate_func=linear)
        self.play(Rotating(origin, about_point=[2, 0, 0]), run_time=2)
        self.wait()
        self.play(FadeOut(circles))
        self.play(FadeOut(origin))
