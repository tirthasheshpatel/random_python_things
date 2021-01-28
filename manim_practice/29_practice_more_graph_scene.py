import numpy as np

from manim import *

class ALittleComplexScene(Scene):
    def construct(self):
        self.all_objects = VGroup()
        self.create_axes()
        self.create_circle()
        self.create_sin_graph()

    def create_axes(self):
        self.origin_point = np.array([-4, 0, 0])
        self.x_axis = Line(self.origin_point + 2*LEFT,
                           self.origin_point + 10*RIGHT)
        self.y_axis = Line(self.origin_point + 2*DOWN,
                           self.origin_point + 2*UP)
        self.x_axis_labels = VGroup(
            MathTex(r"1\pi"), MathTex(r"2\pi"),
            MathTex(r"3\pi"), MathTex(r"4\pi")
        )
        for i, lab in enumerate(self.x_axis_labels):
            lab.next_to(self.origin_point + RIGHT + 2*(i+1)*RIGHT,
                        DOWN)
        self.play(ShowCreation(self.x_axis))
        self.play(ShowCreation(self.y_axis))
        self.play(Write(self.x_axis_labels))
        self.wait()
        self.all_objects.add(self.x_axis, self.y_axis,
                             self.x_axis_labels)

    def create_circle(self):
        self.circle = Circle(radius=1).move_to(self.origin_point)
        self.play(ShowCreation(self.circle))
        self.wait()
        self.all_objects.add(self.circle)

    def create_sin_graph(self):
        orbit = self.circle
        origin_point = self.origin_point
        curve_point = np.array([-3, 0, 0])

        tracing_point = Dot(radius=0.08, color=YELLOW)
        tracing_point.move_to(orbit.point_from_proportion(0))
        line_to_circle = Line(origin_point,
                              tracing_point.get_center(),
                              color=BLUE)
        line_to_curve  = Line(tracing_point.get_center(),
                              tracing_point.get_center(),
                              color=YELLOW_A,
                              stroke_width=2)
        curve = VMobject().set_color(YELLOW_D)
        curve.set_points_as_corners([curve_point, curve_point])

        self.play(ShowCreation(line_to_circle))
        self.play(ShowCreation(tracing_point))
        self.add(line_to_curve, curve)

        self.t_offset = 0
        rate = 0.25

        def go_around_circle(mobj: Dot, dt: float) -> None:
            self.t_offset += (dt * rate)
            mobj.move_to(
                orbit.point_from_proportion(self.t_offset % 1)
            )

        def get_line_to_circle() -> Line:
            return Line(origin_point,
                        tracing_point.get_center(),
                        color=BLUE,
                        z_index=-1)

        def get_line_to_curve(mobj: Line) -> None:
            x = curve_point[0] + self.t_offset * 1/rate
            y = tracing_point.get_center()[1]
            mobj.set_points_as_corners(
                [tracing_point.get_center(),
                 np.array([x, y, 0])]
            )

        def get_curve(mobj: VMobject) -> None:
            x = curve_point[0] + self.t_offset * 1/rate
            y = tracing_point.get_center()[1]
            new_point = np.array([x, y, 0])
            mobj.add_points_as_corners([new_point])

        tracing_point.add_updater(go_around_circle)
        line_to_curve.add_updater(get_line_to_curve)
        curve.add_updater(get_curve)

        line_to_circle_artist = always_redraw(get_line_to_circle)
        self.add(line_to_circle_artist)
        self.remove(line_to_circle)

        self.wait(1/rate * 2 + 1e-3)

        tracing_point.remove_updater(go_around_circle)
        line_to_curve.remove_updater(get_line_to_curve)
        curve.remove_updater(get_curve)

        self.wait()
        self.all_objects.add(tracing_point,
                             line_to_circle_artist,
                             line_to_curve,
                             curve)

        self.play(Uncreate(self.all_objects))
        self.wait()
