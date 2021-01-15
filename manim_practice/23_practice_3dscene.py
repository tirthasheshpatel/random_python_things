import numpy as np
from manim import *

FRAME_WIDTH = config.frame_width
FRAME_HEIGHT = config.frame_height
PIXEL_WIDTH = config.pixel_width
PIXEL_HEIGHT = config.pixel_height

class ThreeDSceneInManim(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        text = Tex("This is 3D Text!!!").to_corner(UL).set_opacity(0)
        self.play(ShowCreation(axes))
        self.add_fixed_in_frame_mobjects(text)
        self.play(text.animate.set_opacity(1))
        self.wait()
