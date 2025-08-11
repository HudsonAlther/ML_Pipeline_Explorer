# random_forest_training.py
from manim import *
import random

class RandomForestTraining(Scene):
    """Show 3 mini decision-trees and highlight a prediction path."""

    def mini_tree(self, root_pos: np.ndarray) -> VGroup:
        """Return a VGroup containing nodes and edges for a simple depth-2 tree."""
        # Node style
        node_r = 0.15
        node_color = WHITE

        # Create nodes
        root = Circle(node_r, color=node_color).move_to(root_pos)
        left = Circle(node_r, color=node_color).next_to(root, DOWN + LEFT, buff=0.6)
        right = Circle(node_r, color=node_color).next_to(root, DOWN + RIGHT, buff=0.6)
        leaf_l = Circle(node_r, color=node_color).next_to(left, DOWN, buff=0.6)
        leaf_r = Circle(node_r, color=node_color).next_to(right, DOWN, buff=0.6)

        # Edges
        edges = VGroup(
            Line(root.get_bottom(), left.get_top()),
            Line(root.get_bottom(), right.get_top()),
            Line(left.get_bottom(), leaf_l.get_top()),
            Line(right.get_bottom(), leaf_r.get_top()),
        )

        nodes = VGroup(root, left, right, leaf_l, leaf_r)
        return VGroup(edges, nodes)

    def construct(self):
        tree_group = VGroup()
        for i, x_shift in enumerate([-3, 0, 3]):  # three trees horizontally, balanced
            tree = self.mini_tree(np.array([x_shift, 2, 0]))
            tree_group.add(tree)
            self.play(Create(tree), run_time=1)

            # Simulate training by flashing nodes randomly
            # nodes order: root, left, right, leaf_l, leaf_r (see mini_tree construction)
            nodes_list = list(tree[1])
            root_node, left_node, right_node, leaf_l, leaf_r = nodes_list

            if i == 0:  # first (leftmost) tree – highlight root + left branch
                self.play(*[Indicate(n, scale_factor=1.3) for n in [root_node, left_node, leaf_l]], run_time=1)
            elif i == 1:  # middle tree – highlight root + right branch
                self.play(*[Indicate(n, scale_factor=1.3) for n in [root_node, right_node, leaf_r]], run_time=1)
            else:  # third tree – highlight root + both children
                self.play(*[Indicate(n, scale_factor=1.4) for n in [root_node, left_node, right_node]], run_time=1)

        self.wait(1)
