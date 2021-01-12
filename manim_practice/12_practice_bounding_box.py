from manim import *

FRAME_WIDTH = config.frame_width
FRAME_HEIGHT = config.frame_height
PIXEL_WIDTH = config.pixel_width
PIXEL_HEIGHT = config.pixel_height

class BoundingBoxInManim(Scene):
    def construct(self):
        baye = MathTex(r"P(\theta \mid X)", r"=",
                       r"\frac{P(X \mid \theta)P(\theta)}{P(X)}")

        self.wait()
        self.play(Write(baye))
        boundingbox1 = SurroundingRectangle(baye[0])
        boundingbox2 = SurroundingRectangle(baye[2])
        self.play(ShowCreation(boundingbox1))
        self.wait(0.2)
        self.play(ReplacementTransform(boundingbox1, boundingbox2))
        self.wait(0.2)
        self.play(FadeOut(boundingbox2))
        self.play(baye.animate.scale(0))
        self.remove(baye)
        self.wait()
