from manim import *

FRAME_WIDTH = config.frame_width
FRAME_HEIGHT = config.frame_height
PIXEL_WIDTH = config.pixel_width
PIXEL_HEIGHT = config.pixel_height

class PlacingStuff(Scene):
    def construct(self) -> None:
        ax1 = Line( FRAME_HEIGHT * DOWN, FRAME_HEIGHT * UP    ).to_edge(LEFT)
        ax2 = Line( FRAME_WIDTH  * LEFT, FRAME_WIDTH  * RIGHT ).to_edge(DOWN)
        axis_frame = Rectangle(height=FRAME_HEIGHT, width=FRAME_WIDTH)

        ax1_xy = ax1.get_x(), ax1.get_y()
        ax2_xy = ax2.get_x(), ax2.get_y()
        ax1.next_to(axis_frame, DOWN)
        ax2.next_to(axis_frame, LEFT)
        ax_ = VGroup(ax1, ax2).set_fill(opacity=0)

        ax1.set_x(ax1_xy[0])
        ax2.set_y(ax2_xy[1])

        def show_axis_creation(ax: VGroup) -> VGroup:
            ax1, ax2 = ax
            ax1.set_y(ax1_xy[1])
            ax2.set_x(ax2_xy[0])
            ax_.set_fill(opacity=1)
            return ax

        sq1 = Square()
        tr1 = Triangle()
        re1 = Rectangle(height=3, width=5)
        gr1 = VGroup(sq1, tr1, re1)

        self.play(ApplyFunction(show_axis_creation, ax_))
        # self.play(ShowCreation(ax1))
        # self.play(ShowCreation(ax2))
        self.wait()
        self.play(ShowCreation(sq1))
        self.wait()
        self.play(ApplyMethod(sq1.move_to, 2*LEFT))
        self.play(ShowCreation(tr1.next_to(sq1, RIGHT)))
        self.play(ShowCreation(re1.align_to(tr1, RIGHT)))
        self.wait()
        self.play(FadeOut(gr1))
        self.play(FadeOut(ax_))
