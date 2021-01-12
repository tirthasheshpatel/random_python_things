from manim import *


class GroupExample(Scene):
    def construct(self):
        c1 = Circle().shift(LEFT)
        r1 = Rectangle().shift(RIGHT)
        g1 = VGroup(c1, r1)

        c2 = Circle().shift(RIGHT + 2 * DOWN)
        s2 = Square().shift(LEFT + 2 * DOWN)
        r2 = Rectangle(height=4, width=2).shift(2 * DL)
        g2 = VGroup(r2, c2, s2)

        self.play(ShowCreation(g1))
        self.play(ReplacementTransform(g1, g2))
        self.play(FadeOut(c2))
        self.play(FadeOut(s2))
        self.play(FadeOut(r2))
        self.wait()
