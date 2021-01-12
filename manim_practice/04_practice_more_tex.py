from manim import *


class WriteStuff(Scene):
    def construct(self):
        text = Tex("I love Baye!", tex_to_color_map={"Baye": YELLOW})
        tex = MathTex(
            r"P(\theta \mid X) = " r"\frac{P(X \mid \theta)" r"P(\theta)}{P(X)}",
            # tex_to_color_map={"\\theta": GREEN,
            #                   "X": ORANGE}
        )
        group = VGroup(text, tex)
        group.arrange(DOWN)
        group.set_width(config.frame_width - 2 * LARGE_BUFF)

        self.play(Write(text))
        self.play(Write(tex))
        self.wait()
