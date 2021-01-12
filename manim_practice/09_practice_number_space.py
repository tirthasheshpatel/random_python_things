from manim import *

FRAME_WIDTH = config.frame_width
FRAME_HEIGHT = config.frame_height
PIXEL_WIDTH = config.pixel_width
PIXEL_HEIGHT = config.pixel_height

class PlayingWithNumberSpace(Scene):
    def construct(self):
        origin       = Dot(ORIGIN)
        vector       = Arrow(origin, [2, 2, 0], buff=0)
        number_plane = NumberPlane()
        origin_label = MathTex("(0, 0)").next_to(origin, DOWN)
        vector_label = MathTex("(2, 2)").next_to(vector, UP)

        all_objs     = VGroup(origin, vector, number_plane,
                              origin_label, vector_label)

        self.wait()
        self.wait()
        self.play(ShowCreation(VGroup(number_plane, origin, origin_label)))
        self.wait()
        self.play(ShowCreation(VGroup(vector, vector_label)))
        self.wait()

        self.play(FadeOut(all_objs))
