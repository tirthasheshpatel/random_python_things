import numpy as np
from manim import *

class Rotating3DScene(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        circle = Circle()
        self.set_camera_orientation(phi=75*DEGREES, theta=45*DEGREES)
        self.play(ShowCreation(axes))
        self.play(ShowCreation(circle))
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(3)
        self.stop_ambient_camera_rotation()
        self.move_camera(phi=75*DEGREES, theta=45*DEGREES)
        self.wait()
        self.play(Uncreate(circle), Uncreate(axes))
        self.wait()
