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
        action = SymbolicAction(term=-1, action=SymbolicActionEnum.ADD_1)

        next_states = self.regr.next_state(
            states=[start_state],
            actions=[action]
        )[0]

        self.assertEqual(next_states[0], goal_state)

    def test_add_to_entire_expression_with_one_term(self):
        """Test adding and subtracting to an expression using -1 term index, where the expression is only one term."""
        actions = [
            SymbolicActionEnum.ADD_1,
            SymbolicActionEnum.ADD_X,
            SymbolicActionEnum.SUBTRACT_1,
            SymbolicActionEnum.SUBTRACT_X
        ]
        actions = [SymbolicAction(term=-1, action=action) for action in actions]

        start_state = SymbolicState(expression=x)
        start_states = len(actions) * [start_state]

        goal_states = [
            SymbolicState(x+1),
            SymbolicState(Integer(2) * x),
            SymbolicState(x - 1),
            SymbolicState(0)
        ]

        next_states, costs = self.regr.next_state(start_states, actions)

        for state, goal in zip(next_states, goal_states):
            self.assertEqual(state, goal)

    def test_add_to_entire_expression_with_two_terms(self):
        """Test adding and subtracting to an expression using the -1 term index, where the expression has two terms."""
        actions = [
            SymbolicActionEnum.ADD_1,
            SymbolicActionEnum.ADD_X,
            SymbolicActionEnum.SUBTRACT_1,
            SymbolicActionEnum.SUBTRACT_X
        ]
        actions = [SymbolicAction(term=-1, action=action) for action in actions]

        start_state = SymbolicState(expression=x+1)
        start_states = len(actions) * [start_state]

        goal_states = [
            SymbolicState(x+2),
            SymbolicState(Integer(2) * x + 1),
            SymbolicState(x),
            SymbolicState(1)
        ]

        next_states, costs = self.regr.next_state(start_states, actions)

        for state, goal in zip(next_states, goal_states):
            self.assertEqual(state, goal)

    def test_add_to_first_term(self):
        """Test adding and subtracting values to the first term in the expression."""
        actions = [
            SymbolicActionEnum.ADD_1,
            SymbolicActionEnum.ADD_X,
            SymbolicActionEnum.SUBTRACT_1,
            SymbolicActionEnum.SUBTRACT_X
        ]
        actions = [SymbolicAction(term=0, action=action) for action in actions]

        start_state = SymbolicState(expression=x+1)
        start_states = len(actions) * [start_state]

        # The terms are not always in the order as shown in the expression
        # Term 0 corresponds to the "1" in "x + 1"
        goal_states = [
            SymbolicState(x+2),
            SymbolicState(Integer(2) * x + 1),
            SymbolicState(x),
            SymbolicState(1)
        ]

        next_states, costs = self.regr.next_state(start_states, actions)

        for state, goal in zip(next_states, goal_states):
            self.assertEqual(state, goal)

    def test_multiply_entire_expression(self):
        """Test multiplying and dividing the entire expression."""
        actions = [
            SymbolicActionEnum.MULTIPLY_BY_2,
            SymbolicActionEnum.MULTIPLY_BY_X,
            SymbolicActionEnum.DIVIDE_BY_2,
            SymbolicActionEnum.DIVIDE_BY_X
        ]
        actions = [SymbolicAction(term=-1, action=action) for action in actions]

        start_state = SymbolicState(expression=x + 1)
        start_states = len(actions) * [start_state]

        goal_states = [
            SymbolicState(2*x + 2),
            SymbolicState(x**2 + x),
            SymbolicState(x/2 + 1/2),
            SymbolicState(1 + 1/x)
        ]

        next_states, costs = self.regr.next_state(start_states, actions)

        for state, goal in zip(next_states, goal_states):
            self.assertEqual(state, goal)

    def test_multiply_first_term(self):
        """Test multiplying the first term, 0th index operates on the "1" in "x+1"."""
        actions = [
            SymbolicActionEnum.MULTIPLY_BY_2,
            SymbolicActionEnum.MULTIPLY_BY_X,
            SymbolicActionEnum.DIVIDE_BY_2,
            SymbolicActionEnum.DIVIDE_BY_X
        ]
        actions = [SymbolicAction(term=0, action=action) for action in actions]

        start_state = SymbolicState(expression=x + 1)
        start_states = len(actions) * [start_state]

        goal_states = [
            SymbolicState(x + 2),
            SymbolicState(2*x),
            SymbolicState(x + 1/2),
            SymbolicState(x + 1/x)
        ]

        next_states, costs = self.regr.next_state(start_states, actions)

        for state, goal in zip(next_states, goal_states):
            self.assertEqual(state, goal)

    def test_multiply_second_term(self):
        """Test multiplying the second term, 1st index operates on the "x" in "x+1"."""
        actions = [
            SymbolicActionEnum.MULTIPLY_BY_2,
            SymbolicActionEnum.MULTIPLY_BY_X,
            SymbolicActionEnum.DIVIDE_BY_2,
            SymbolicActionEnum.DIVIDE_BY_X
        ]
        actions = [SymbolicAction(term=1, action=action) for action in actions]

        start_state = SymbolicState(expression=x + 1)
        start_states = len(actions) * [start_state]

        goal_states = [
            SymbolicState(2*x + 1),
            SymbolicState(x**2 + 1),
            SymbolicState(x/2 + 1),
            SymbolicState(2)
        ]

        next_states, costs = self.regr.next_state(start_states, actions)

        for state, goal in zip(next_states, goal_states):
            self.assertEqual(state, goal)

    def test_multiply_by_negative_one(self):
        """Test multiplying by negative 1."""
        actions = [SymbolicAction(term=-1, action=SymbolicActionEnum.MULTIPLY_BY_NEG_1)]

        start_state = SymbolicState(expression=x + 1)
        start_states = len(actions) * [start_state]

        goal_states = [
            SymbolicState(-x - 1)
        ]

        next_states, costs = self.regr.next_state(start_states, actions)

        for state, goal in zip(next_states, goal_states):
            self.assertEqual(state, goal)
