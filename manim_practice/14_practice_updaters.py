from numpy import ndarray
from manim import *

FRAME_WIDTH = config.frame_width
FRAME_HEIGHT = config.frame_height
PIXEL_WIDTH = config.pixel_width
PIXEL_HEIGHT = config.pixel_height

class MobjectUpdaterInManim(Scene):
    def construct(self):
        l1 = Line(ORIGIN, LEFT)
        l2 = Line(ORIGIN, LEFT).set_color(ORANGE)

        # counter clockwise rotation (ccr) updater
        def ccr_updater(mobj: Mobject, dt: ndarray) -> Mobject:
            mobj.rotate(dt, about_point=ORIGIN)
            return mobj

        # clockwise rotation (cr) updater
        def cr_updater(mobj, dt):
            mobj.rotate(-dt, about_point=ORIGIN)

        self.wait()
        self.play(ShowCreation(VGroup(l1, l2)))
        l2.add_updater(ccr_updater)
        self.wait(2)
        l2.remove_updater(ccr_updater)
        l2.add_updater(cr_updater)
        self.wait(2)
        l2.remove_updater(cr_updater)
        self.wait()
