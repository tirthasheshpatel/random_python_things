import numpy as np
from manim import *

FRAME_WIDTH = config.frame_width
FRAME_HEIGHT = config.frame_height
PIXEL_WIDTH = config.pixel_width
PIXEL_HEIGHT = config.pixel_height

class GraphSceneOvenEngineered(GraphScene):
    def __init__(self, **kwargs):
        GraphScene.__init__(
            self,
            x_min=0,
            x_max=40,
            y_min=-7,
            y_max=30,
            x_labeled_nums=range(0, 45, 5),
            y_labeled_nums=range(-5, 30, 5),
            x_axis_label="$\Delta Q$",
            y_axis_label="$T[^{\circ} C]$",
            **kwargs
        )

    def construct(self):
        self.setup_axes(animate=True)
        self.play(self.axes.animate.scale(0.7))
        y = [20, 0, 0, -5]
        x = [0, 8, 38, 39]

        colors = [RED, BLUE, GREEN, ORANGE]
        points = VGroup()
        lines  = VGroup()
        for a, b in zip(x, y):
            point = Dot().scale(0.7).move_to(self.coords_to_point(a, b))
            points.add(point)
        for i in range(1, len(points)):
            line = Line(points[i-1], points[i])
            lines.add(line)
        labels = [MathTex(f"x = {i}") for i in x]
        angles = [5 * PI / 4, 3 * PI / 2, PI / 4, PI / 2]
        sides  = [UR*0.5, UP*0.6, DL*0.5, DOWN*0.6]
        arrows = [
            Arrow(buff=SMALL_BUFF, preserve_tip_size_when_scaling=False,
                  tip_length=0.2)
            .scale(0.5)
            .move_to(points[i]
                     .get_center() + sides[i])
            .rotate(angles[i])
            for i in range(len(x))
        ]
        labels_sides = [UR*0.7, UP*0.7, DL*0.7, DOWN*0.7]
        labels = [
            labels[i]
            .scale(0.7)
            .next_to(arrows[i], labels_sides[i])
            for i in range(len(x))
        ]
        arrows = VGroup(*arrows)
        labels = VGroup(*labels)
        for i in range(len(x)):
            points[i].set_color(colors[i])
            if i!=len(x)-1: lines[i].set_color(colors[i])
            arrows[i].set_color(colors[i])
            labels[i].set_color(colors[i])

        all_objs = VGroup(self.axes, points, lines, arrows, labels)

        self.play(ShowCreation(points))
        for line in lines:
            self.play(ShowCreation(line))
        for arrow, label in zip(arrows, labels):
            self.play(ShowCreation(arrow), ShowCreation(label))
        self.wait()
        self.play(FadeOut(all_objs))
        self.wait()
