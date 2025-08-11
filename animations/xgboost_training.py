# xgboost_training.py
from manim import *


class XGBoostTraining(Scene):
    """Boosting illustration: initial guess, error shown, tree, improved guess bar."""

    def construct(self):
        # Build bar from two halves so we can recolor top part
        bottom_half = Rectangle(width=1.0, height=1.0, fill_color=GREY, fill_opacity=0.8)
        top_half = Rectangle(width=1.0, height=1.0, fill_color=GREY, fill_opacity=0.8)
        top_half.next_to(bottom_half, UP, buff=0)
        guess_bar = VGroup(bottom_half, top_half)

        guess_label = Text("Initial guess").scale(0.35).next_to(top_half, UP, buff=0.1)
        self.play(GrowFromEdge(guess_bar, edge=DOWN), FadeIn(guess_label))
        self.wait(0.6)

        # Fade out label
        self.play(FadeOut(guess_label))

        # Top half becomes red (no overlay)
        self.play(top_half.animate.set_fill(RED, opacity=0.85), run_time=0.7)
        self.wait(0.6)

        # Grow tiny tree
        root = Circle(0.25, color=WHITE).shift(LEFT * 2.5 + UP * 0.5)
        left_leaf = Circle(0.18, color=WHITE).next_to(root, DOWN + LEFT, buff=0.4)
        right_leaf = Circle(0.18, color=WHITE).next_to(root, DOWN + RIGHT, buff=0.4)
        edges = VGroup(Line(root.get_bottom(), left_leaf.get_top()), Line(root.get_bottom(), right_leaf.get_top()))
        self.play(Create(VGroup(root, left_leaf, right_leaf, edges)), run_time=1.5)
        self.wait(0.5)

        # New green bar to the right with height equal to bottom half (correction amount)
        green_bar = Rectangle(width=1.0, height=1.0, fill_color=GREEN, fill_opacity=0.8)
        green_bar.next_to(bottom_half, RIGHT, buff=1.2)
        green_bar.align_to(bottom_half, DOWN)
        improved_label = Text("Improved guess").scale(0.35).next_to(green_bar, UP, buff=0.1)
        self.play(GrowFromEdge(green_bar, edge=DOWN), FadeIn(improved_label))
        self.wait(1.5)
