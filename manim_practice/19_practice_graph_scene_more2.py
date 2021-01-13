import numpy as np
from numpy import ndarray
from manim import *

def temp_func(x: ndarray) -> ndarray:
    return (x <= 8.) * (20. - 5.*x / 2.) - (x >= 38.) * (x-38) * 1.5

class YetAnotherGraphScene(GraphScene):
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
            y_axis_label="$T[^{\circ} C]$"
        )

    def construct(self):
        self.setup_axes(animate=True)

        curve    = self.get_graph(temp_func, color=RED)
        all_objs = VGroup(curve, self.axes)

        self.play(ShowCreation(curve))
        self.wait()
        self.play(FadeOut(all_objs))
        self.wait()
