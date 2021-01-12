from typing import List
import numpy as np

from manim import *

FRAME_WIDTH = config.frame_width
FRAME_HEIGHT = config.frame_height
PIXEL_WIDTH = config.pixel_width
PIXEL_HEIGHT = config.pixel_height

class UsingImagesInManim(Scene):
    def construct(self):
        m, n = 480, 640
        img = np.r_[ m*[np.arange(n, dtype=np.uint8)] ]
        img_mobj = ImageMobject(img)

        tirth = ImageMobject("images/tirth.jpg")

        self.wait(2)
        self.play(FadeIn(img_mobj))
        self.wait()
        self.play(ReplacementTransform(img_mobj, tirth))
        self.play(ApplyMethod(tirth.scale, 0))
        self.remove(tirth)
        self.wait()
