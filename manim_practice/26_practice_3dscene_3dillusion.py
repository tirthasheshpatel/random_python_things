import numpy as np
from manim import *

class ThreeDIllusionScene(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        circle = Circle()
        self.set_camera_orientation(phi=75*DEGREES, theta=-60*DEGREES)
        self.play(ShowCreation(axes), ShowCreation(circle))
        self.wait()
        self.begin_3dillusion_camera_rotation(rate=2)
        self.wait(2*PI)
        self.stop_3dillusion_camera_rotation()
        self.wait()
        self.play(Uncreate(axes))
        self.wait()
