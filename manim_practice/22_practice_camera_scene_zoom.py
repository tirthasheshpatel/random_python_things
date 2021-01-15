import numpy as np
from manim import *

FRAME_WIDTH = config.frame_width
FRAME_HEIGHT = config.frame_height
PIXEL_WIDTH = config.pixel_width
PIXEL_HEIGHT = config.pixel_height

class CameraSceneWithZooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooom(ZoomedScene):
    def __init__(self, **kwargs):
        ZoomedScene.__init__(
            self,
            zoomed_display_height=1,
            zoomed_display_width=6,
            zoom_factor=0.3,
            image_frame_stroke_width=20,
            zoomed_camera_config={
                "default_frame_stroke_width": 3,
            },
            **kwargs
        )

    def construct(self):
        point = Dot().shift(2 * UL)
        image = ImageMobject(np.uint8([[0, 100, 30, 200],
                                       [255, 0, 5, 33]]))
        image.set_height(7)

        self.play(FadeIn(image), ShowCreation(point))

        zoomed_camera = self.zoomed_camera
        zoomed_display = self.zoomed_display

        zoomed_camera_frame = zoomed_camera.frame
        zoomed_camera_frame_text = Tex("Zoomed Camera Frame", color=ORANGE)

        zoomed_display_frame = zoomed_display.display_frame
        zoomed_display_frame_text = Tex("Zoomed Display Frame", color=PURPLE)

        zoomed_camera_frame.move_to(point)
        zoomed_camera_frame.set_color(ORANGE)
        zoomed_camera_frame_text.next_to(zoomed_camera_frame, DOWN)

        zoomed_display_frame.set_color(PURPLE)
        zoomed_display.shift(DOWN)

        zoomed_display_bounding_box = BackgroundRectangle(
            zoomed_display, fill_opacity=0, buff=MED_SMALL_BUFF
        )
        self.add_foreground_mobject(zoomed_display_bounding_box)

        unfold_camera = UpdateFromFunc(
            zoomed_display_bounding_box,
            lambda box: box.replace(zoomed_display)
        )
        self.play(
            ShowCreation(zoomed_camera_frame),
            FadeInFrom(zoomed_camera_frame_text, direction=DOWN)
        )
        self.activate_zooming()
        self.play(
            self.get_zoomed_display_pop_out_animation(),
            unfold_camera
        )
        zoomed_display_frame_text.next_to(zoomed_display_frame, DOWN)
        self.play(FadeInFrom(zoomed_display_frame_text, direction=DOWN))
        scale_factor = [0.5, 1.5, 0]
        self.play(
            zoomed_camera_frame.animate.scale(scale_factor),
            zoomed_display.animate.scale(scale_factor),
            FadeOut(zoomed_camera_frame_text),
            FadeOut(zoomed_display_frame_text)
        )
        self.wait()
        self.play(ScaleInPlace(zoomed_display, 2))
        self.wait()
        self.play(zoomed_camera_frame.animate.shift(2.5 * DOWN))
        self.play(zoomed_camera_frame.animate.shift(2.5 * UP))
        self.wait()
        self.play(
            self.get_zoomed_display_pop_out_animation(),
            unfold_camera, rate_func=lambda dt: smooth(1-dt)
        )
        self.play(Uncreate(zoomed_display_frame),
                  FadeOut(zoomed_camera_frame))
        self.wait()
