# TODO: practice 3D scene more
# TODO: create a cool animation in 3D

from scipy.stats import multivariate_normal
import numpy as np
from manim import *

FRAME_WIDTH = config.frame_width
FRAME_HEIGHT = config.frame_height
PIXEL_WIDTH = config.pixel_width
PIXEL_HEIGHT = config.pixel_height

class Cool3DScene(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(z_min=-0.1, z_max=0.3,
                          z_axis_config={"preserve_tip_size_when_scaling": False,
                                         "tip_length": 0.08},
                          x_axis_config={"x_min": -2,
                                         "x_max": +2,
                                         "preserve_tip_size_when_scaling": False,
                                         "tip_length": 0.2},
                          y_axis_config={"x_min": -2,
                                         "x_max": +2,
                                         "preserve_tip_size_when_scaling": False,
                                         "tip_length": 0.2})
        axes.scale([2, 2, 12])

        def normal_pdf(x, y):
            return np.array([
                x, y,
                multivariate_normal.pdf([x, y], np.zeros(2), np.eye(2))
            ])
        normal_surface = ParametricSurface(
            normal_pdf,
            u_min=-2, u_max=2,
            v_min=-2, v_max=2,
            resolution=16
        ).scale_about_point([2, 2, 12], axes.get_center())
        normal_surface.set_style(fill_opacity=1)
        normal_surface.set_style(stroke_color=RED)
        normal_surface.set_fill_by_checkerboard(RED_D, RED_E)
        self.set_camera_orientation(phi=75*DEGREES, theta=45*DEGREES)
        self.wait()
        self.play(Write(axes))
        self.begin_ambient_camera_rotation(rate=0.05)
        self.wait(2)
        self.play(Write(normal_surface))
        self.wait(2)
        self.play(normal_surface.animate.set_fill_by_checkerboard(RED_D, RED_E, opacity=0.1))
        self.play(normal_surface.animate.set_style(stroke_width=0))
        self.wait(3)
        self.stop_ambient_camera_rotation()
        self.play(FadeOut(VGroup(normal_surface, axes)))
        self.wait()
