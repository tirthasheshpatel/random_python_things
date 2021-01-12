from manim import *


class TexExhibit(Scene):
    def construct(self):
        tex1 = MathTex(r"\sum_{i=0}^{n} i = \frac{n(n+1)}{2}")
        tex2 = MathTex(r"\sum_{i=0}^{n} i^2 = \frac{n}{6}(n+1)(2n+1)")
        tex3 = MathTex(r"\sum_{i=0}^{n} i^3 = \frac{n^2(n+1)^2}{4}")
        text1 = Tex("The sum of different powers!")
        self.play(Write(text1))
        self.wait()
        # self.play(
        #     Homotopy(
        #         lambda x, y, z, t: (x, y - 0.2 * config.frame_height * t, z), text1
        #     ).set_run_time(1)
        # )
        self.play(ApplyMethod(text1.shift, 2 * UP))
        self.play(ShowCreation(tex1))
        self.wait()
        self.play(ReplacementTransform(tex1, tex2))
        self.wait()
        self.play(ReplacementTransform(tex2, tex3))
        self.wait()
        self.play(FadeOut(tex3))
        self.play(FadeOut(text1))
