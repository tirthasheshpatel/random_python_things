from manim import *


class WhatCanDo(Scene):
    def construct(self):
        text = VGroup(Tex("Chapter 1"), Tex("Baye's Theorem"))
        text.arrange(DOWN).scale(2.5).set_fill(opacity=0)
        t1_y = text[0].get_y()
        t2_y = text[1].get_y()
        screen = Rectangle(width=config.frame_width, height=config.frame_height)

        for pos, t in zip([UP, DOWN], text):
            t.next_to(screen, pos, buff=SMALL_BUFF)

        def show_text(text_):
            t1, t2 = text_
            t1.set_y(t1_y)
            t2.set_y(t2_y)
            text_.set_fill(opacity=1)
            return text_

        def disappear_text(text_):
            for pos, t in zip([RIGHT, LEFT], text_):
                t_c = t.copy()
                t.next_to(screen, pos, buff=SMALL_BUFF)
                t.set_y(t_c.get_y())
                t.set_fill(opacity=0)
            return text_

        self.play(ApplyFunction(show_text, text))
        self.wait()
        self.play(ApplyFunction(disappear_text, text))
        self.wait()
