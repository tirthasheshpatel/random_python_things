from manim import *

FRAME_WIDTH = config.frame_width
FRAME_HEIGHT = config.frame_height
PIXEL_WIDTH = config.pixel_width
PIXEL_HEIGHT = config.pixel_height

class PlaceLogo(Scene):
    def construct(self):
        self.camera.background_color = "#ece6e2"
        logo_green = "#87c2a5"
        logo_blue = "#525893"
        logo_red = "#e07a5f"
        logo_black = "#343434"

        tr1 = Triangle(color=logo_red).set_fill(color=ORANGE, opacity=1).set_stroke(width=0)
        sq1 = Square(color=logo_blue).set_fill(color=PURPLE, opacity=1).set_stroke(width=0)
        ci1 = Circle(color=logo_green).set_fill(color=GREEN, opacity=1).set_stroke(width=0)
        te1 = MathTex(r"\mathbb{M}", fill_color=logo_black)
        all_objs = VGroup(tr1, sq1, ci1, te1)

        self.play(ShowCreation(tr1), run_time=0.5)
        self.play(ApplyMethod(tr1.move_to, DOWN + 2*RIGHT), run_time=0.5)
        self.play(ShowCreation(sq1), run_time=0.5)
        self.play(ApplyMethod(sq1.move_to, RIGHT), run_time=0.5)
        self.play(ShowCreation(ci1), run_time=0.5)
        self.play(ApplyMethod(ci1.move_to, DOWN), run_time=0.5)
        self.play(ShowCreation(te1), run_time=0.5)
        self.play(ApplyMethod(te1.scale, 7), run_time=0.5)
        self.play(ApplyMethod(te1.move_to, 0.5 * UP + 1.25 * LEFT), run_time=0.5)
        self.wait()
        self.play(FadeOut(all_objs))
        self.wait()
