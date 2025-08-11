# logistics_training.py
from manim import *
import numpy as np
import random

class LogisticTraining(Scene):
    """Simple logistic-regression separator animation."""
    def construct(self):
        # Scatter blue vs red points
        class0 = [Dot(point=[x, -1 + 0.2 * random.random(), 0], color=BLUE) for x in np.linspace(-4, 0, 20)]
        class1 = [Dot(point=[x,  1 - 0.2 * random.random(), 0], color=RED)  for x in np.linspace(0,  4, 20)]
        self.play(*[Create(d) for d in class0 + class1])
        self.wait(0.5)

        # Decision boundary initial line
        line = Line([-5, 0, 0], [5, 0, 0], color=YELLOW)
        self.add(line)

        # Rotate to mimic training improving separator
        self.play(Rotate(line, angle=PI / 6, about_point=ORIGIN), run_time=2)
        self.play(Rotate(line, angle=-PI / 12, about_point=ORIGIN), run_time=2)
        self.wait(1)
