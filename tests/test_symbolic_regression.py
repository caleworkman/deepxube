import unittest

from sympy import *
from sympy.abc import x

from deepxube.domains.symbolic_regression import SymbolicAction, SymbolicActionEnum, SymbolicState, SymbolicRegression


class TestSymbolicRegression(unittest.TestCase):
    # python - m unittest tests.test_symbolic_regression.TestSymbolicRegression

    def setUp(self):
        self.regr = SymbolicRegression(random_walk_length=5)

    def test_add_one_to_x(self):

        start_state = SymbolicState(expression=x)
        goal_state = SymbolicState(expression=x+1)

        # Add 1 to the entire expression
        action = SymbolicAction(
            term=-1,
            action=SymbolicActionEnum.ADD,
            value=1
        )

        next_states = self.regr.next_state(
            states=[start_state],
            actions=[action]
        )[0]

        self.assertEqual(next_states[0], goal_state)

    def test_add_to_entire_expression_with_one_term(self):
        """Test adding to an expression using the -1 term index, where the expression is only one term."""
        values_to_add = [1, 2, 1/2, x]
        goal_states = [
            SymbolicState(x+1),
            SymbolicState(x+2),
            SymbolicState(x+0.5),
            SymbolicState(Integer(2) * x)
        ]

        start_state = SymbolicState(expression=x)
        start_states = len(values_to_add) * [start_state]
        actions = [SymbolicAction(
            term=-1,
            action=SymbolicActionEnum.ADD,
            value=value
        ) for value in values_to_add]

        next_states, costs = self.regr.next_state(start_states, actions)

        for state, goal in zip(next_states, goal_states):
            self.assertEqual(state, goal)

    def test_add_to_entire_expression_with_two_terms(self):
        """Test adding to an expression using the -1 term index, where the expression as two terms."""
        values_to_add = [1, 2, 1/2, x]
        goal_states = [
            SymbolicState(x+2),
            SymbolicState(x+3),
            SymbolicState(x+1.5),
            SymbolicState(Integer(2) * x + 1)
        ]

        start_state = SymbolicState(expression=x+1)
        start_states = len(values_to_add) * [start_state]
        actions = [SymbolicAction(
            term=-1,
            action=SymbolicActionEnum.ADD,
            value=value
        ) for value in values_to_add]

        next_states, costs = self.regr.next_state(start_states, actions)

        for state, goal in zip(next_states, goal_states):
            self.assertEqual(state, goal)

    def test_add_to_first_term(self):
        """Test adding values to the first term in the expression."""

        values_to_add = [1, 2, 1/2, x]
        goal_states = [
            SymbolicState(x+2),
            SymbolicState(x+3),
            SymbolicState(x+1.5),
            SymbolicState(Integer(2) * x + 1)
        ]

        start_state = SymbolicState(expression=x+1)
        start_states = len(values_to_add) * [start_state]

        # The terms are not always in the order as shown in the expression
        # Term 0 corresponds to the "1" in "x + 1"
        actions = [SymbolicAction(
            term=0,
            action=SymbolicActionEnum.ADD,
            value=value
        ) for value in values_to_add]

        next_states, costs = self.regr.next_state(start_states, actions)

        for state, goal in zip(next_states, goal_states):
            self.assertEqual(state, goal)

    def test_multiply_entire_expression(self):
        """Test multiplying the entire expression."""

        values_to_mul = [1, 2, 1 / 2, x]
        goal_states = [
            SymbolicState(x + 1),
            SymbolicState(2 * x + 2),
            SymbolicState(0.5 * x + 0.5),
            SymbolicState(x**2 + x)
        ]

        start_state = SymbolicState(expression=x + 1)
        start_states = len(values_to_mul) * [start_state]

        actions = [SymbolicAction(
            term=-1,
            action=SymbolicActionEnum.MULTIPLY,
            value=value
        ) for value in values_to_mul]

        next_states, costs = self.regr.next_state(start_states, actions)

        for state, goal in zip(next_states, goal_states):
            self.assertEqual(state, goal)

    def test_multiply_first_term(self):
        """Test multiplying the first term, 0th index operates on the "1" in "x+1"."""

        values_to_mul = [1, 2, 1 / 2, x]
        goal_states = [
            SymbolicState(x + 1),
            SymbolicState(x + 2),
            SymbolicState(x + 0.5),
            SymbolicState(2*x)
        ]

        start_state = SymbolicState(expression=x + 1)
        start_states = len(values_to_mul) * [start_state]

        actions = [SymbolicAction(
            term=0,
            action=SymbolicActionEnum.MULTIPLY,
            value=value
        ) for value in values_to_mul]

        next_states, costs = self.regr.next_state(start_states, actions)

        for state, goal in zip(next_states, goal_states):
            self.assertEqual(state, goal)

    def test_multiply_second_term(self):
        """Test multiplying the second term, 1st index operates on the "x" in "x+1"."""

        values_to_mul = [1, 2, 1 / 2, x]
        goal_states = [
            SymbolicState(x + 1),
            SymbolicState(2*x + 1),
            SymbolicState(0.5 * x + 1),
            SymbolicState(x**2 + 1)
        ]

        start_state = SymbolicState(expression=x + 1)
        start_states = len(values_to_mul) * [start_state]

        actions = [SymbolicAction(
            term=1,
            action=SymbolicActionEnum.MULTIPLY,
            value=value
        ) for value in values_to_mul]

        next_states, costs = self.regr.next_state(start_states, actions)

        for state, goal in zip(next_states, goal_states):
            self.assertEqual(state, goal)