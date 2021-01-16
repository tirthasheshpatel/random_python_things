from scipy.stats import multivariate_normal
import numpy as np
from manim import *

class ThreeDPlotting(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        self.set_camera_orientation(phi=75*DEGREES, theta=-30*DEGREES)
        square_plot = ParametricSurface(
            lambda u, v: np.array([u, v, 0]),
            u_min=-2, u_max=2, v_min=-2, v_max=2,
            resolution=(22, 22)
        )
        square_plot.scale_about_point(2, ORIGIN)

        def normal_pdf_func(x, y):
            dist  = np.linalg.norm([x, y])
            mu    = 0
            sigma = 0.4
            z = np.exp( -(dist - mu)**2 / (2 * sigma**2) )
            return np.array([x, y, z])

        normal_pdf = ParametricSurface(
            normal_pdf_func,
            u_min=-2, u_max=2, v_min=-2, v_max=2,
            resolution=(22, 22)
        )
        normal_pdf.scale_about_point(2, ORIGIN)
        normal_pdf.set_style(fill_opacity=1)
        normal_pdf.set_style(stroke_color=GREEN)
        normal_pdf.set_fill_by_checkerboard(GREEN, BLUE, opacity=0.1)

        self.begin_ambient_camera_rotation(rate=0.05)
        self.play(ShowCreation(axes))
        self.wait()
        # wow, `Write` animation has a very serendipitous
        # outcome with ParametricSurfaces!!
        self.play(Write(square_plot))
        self.wait()
        self.play(ReplacementTransform(square_plot, normal_pdf))
        self.wait(3)
        self.stop_ambient_camera_rotation()
        self.play(FadeOut(normal_pdf), FadeOut(axes))
        self.wait()
