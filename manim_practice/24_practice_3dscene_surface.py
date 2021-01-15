import math
import numpy as np
from manim import *

FRAME_WIDTH = config.frame_width
FRAME_HEIGHT = config.frame_height
PIXEL_WIDTH = config.pixel_width
PIXEL_HEIGHT = config.pixel_height

class ThreeDParametricSurfacesInManim(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        unit_sphere = ParametricSurface(
            lambda u, v: np.array([
                math.cos(u) * math.cos(v),
                math.cos(u) * math.sin(v),
                math.sin(u)
            ]), v_min=0, v_max=TAU, u_min=-PI / 2, u_max=PI / 2,
            checkerboard_colors=[RED_D, RED_E], resolution=(16, 32)
        )
        self.renderer.camera.light_source.move_to(3*IN)
        self.set_camera_orientation(phi=75*DEGREES, theta=30*DEGREES)
        self.play(ShowCreation(axes))
        self.play(ShowCreation(unit_sphere))
        self.play(ScaleInPlace(unit_sphere, 1.5))
        self.wait()
        self.play(Uncreate(unit_sphere))
        self.play(FadeOut(axes))
        self.wait()
