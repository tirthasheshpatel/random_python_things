from manim import *

FRAME_WIDTH = config.frame_width
FRAME_HEIGHT = config.frame_height
PIXEL_WIDTH = config.pixel_width
PIXEL_HEIGHT = config.pixel_height

class PutBraces(Scene):
    def construct(self):
        l1 = Line(start=2*LEFT, end=2*RIGHT, fill_color=ORANGE)
        br2 = Brace(l1, direction=UP)
        br2_label = BraceLabel(br2, r"x - x_1",
                               brace_direction=UP,
                               label_constructor=MathTex)
        d1 = Dot().align_to(l1, LEFT)
        d2 = Dot().align_to(l1, RIGHT)
        l_full = VGroup(l1, d1, d2, br2_label)

        self.play(ShowCreation(l_full))
        self.play(Rotate(l_full, PI / 4, about_point=2*LEFT))

        br1 = Brace(l1, DOWN)
        br1_label = BraceLabel(br1, "Horizontal Distance",
                               label_constructor=Tex)

        self.play(ShowCreation(br1_label))

        self.wait()
