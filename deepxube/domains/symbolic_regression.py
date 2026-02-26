from random import randint
from typing import List, Optional, Any

from deepxube.base.factory import Parser
from deepxube.base.domain import State, Action, Goal, ActsEnumFixed, StartGoalWalkable, StringToAct, StateGoalVizable, A
from deepxube.factories.domain_factory import domain_factory

import numpy as np
from sympy import *
from sympy.abc import x

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from deepxube.utils.timing_utils import Times


# pickle file has the start states and goals for examples

class SymbolicState(State):
    def __init__(self, expression: Expr):
        self.expr = expression

    def __hash__(self) -> int:
        return hash(self.expr)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SymbolicState):
            # SymPy tests for equality by subtracting one function from another and testing for 0
            # https://docs.sympy.org/latest/tutorials/intro-tutorial/gotchas.html
            return simplify(self.expr - other.expr) == 0
        return NotImplemented


class SymbolicGoal(Goal):
    def __init__(self, xs: np.typing.NDArray, ys: np.typing.NDArray, tolerance: float):
        """An expression should evaluate to the pairs of values (x, y), eventually implement tolerance."""
        self.xs = xs
        self.ys = ys
        self.tolerance = tolerance


class SymbolicAction(Action):
    def __init__(self, action: int):
        self.action = action

    def __hash__(self) -> int:
        return self.action

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SymbolicAction):
            return self.action == other.action
        return NotImplemented

    def __repr__(self) -> str:
        return f"{self.action}"


@domain_factory.register_class('symbolic_regression')
class SymbolicRegression(
    ActsEnumFixed[SymbolicState, SymbolicAction, SymbolicGoal],
    StartGoalWalkable[SymbolicState, SymbolicAction, SymbolicGoal],
    StateGoalVizable[SymbolicState, SymbolicAction, SymbolicGoal],
    StringToAct[SymbolicState, SymbolicAction, SymbolicGoal]
):

    def __init__(self, random_walk_length: int):
        # Test: python -m deepxube viz --domain symbolic_regression.1
        super().__init__()
        self.random_walk_length = random_walk_length

    def sample_start_states(self, walk_length: int) -> List[SymbolicState]:
        states, costs = self.random_walk([SymbolicState(x)], [walk_length])
        return states

    def sample_goal_from_state(self, states_start: Optional[list[SymbolicState]], states_goal: list[SymbolicState]) -> list[SymbolicGoal]:
        noise = 0  # TODO: Introduce noise later
        xs = np.linspace(0, 1, 20)  # TODO: Change these values later, maybe more points
        xs = np.add(xs, noise)

        goals = []
        for state in states_goal:
            f = lambdify(x, state.expr, 'numpy')
            ys = f(xs)
            goals.append(SymbolicGoal(xs=xs, ys=ys, tolerance=0))
        return goals

    # TODO: How to implement it so the start state and the goal state in the example are different?
    # def sample_start_goal_pairs(self, num_steps_l: list[int], times: Optional[Times] = None) -> tuple[list[SymbolicState], list[SymbolicGoal]]:
    #     print('pairs')
    #     start_states = [SymbolicState(x)]
    #     goals = self.random_walk(start_states, num_steps_l)
    #     return start_states, goals

    def get_actions_fixed(self) -> List[SymbolicAction]:
        return [SymbolicAction(n) for n in range(0, 5)]

    def next_state(self, states: list[SymbolicState], actions: list[SymbolicAction]) -> tuple[list[SymbolicState], list[float]]:
        states_next: List[SymbolicState] = []

        for state, action in zip(states, actions):
            # Terms will be length 0 if there is one term, and >2 if there are more terms. Never 1.
            terms = list(state.expr.args)
            print('terms', terms)
            if len(terms) > 1:
                print('more than 1')
                idx = randint(0, len(terms)-1)
                terms[idx] = self._apply_action(terms[idx], action.action)
                new_expr = state.expr.func(*terms)
            else:
                print('one term')
                new_expr = self._apply_action(state.expr, action.action)

            states_next.append(SymbolicState(new_expr))
        return states_next, [1.0] * len(states_next)

    @staticmethod
    def _apply_action(term, action: int):
        """Apply a sympy manipulation to a term of an expression."""
        if action == 0:
            # Add an integer
            return term + 1
        elif action == 1:
            # Add an x
            return term + x
        elif action == 2:
            # Multiply
            return term * 2
        elif action == 3:
            # Divide
            return term * 1 / 2
        elif action == 4:
            # Increase degree of term (e.g. x -> x^2)
            return term * x
        else:
            raise ValueError('Bad action choice')

    def is_solved(self, states: list[SymbolicState], goals: list[SymbolicGoal]) -> list[bool]:
        # Evaluate each state at the x's in goals, and see if the y's match within some tolerance.
        solved = []
        tolerance = 0

        for state, goal in zip(states, goals):
            f = lambdify(x, state.expr, 'numpy')
            y_values = f(goal.xs)
            within_tolerance = np.less_equal(np.abs(goal.ys - y_values), tolerance)
            solved.append(np.all(within_tolerance))
        return solved

    def string_to_action(self, act_str: str) -> Optional[SymbolicAction]:
        # Just for visualization
        if act_str in {'0', '1', '2', '3', '4'}:
            return SymbolicAction(int(act_str))
        else:
            return None

    def string_to_action_help(self) -> str:
        return ("0 to add 1\n"
                "1 to add x\n"
                "2 to multiply by 2 \n"
                "3 to half\n"
                "4 to multiply by x")

    def visualize_state_goal(self, state: SymbolicState, goal: SymbolicGoal, fig: Figure) -> None:
        # can create a figure with the data points, and the function (state) overlaid
        ax = plt.axes()

        # The goal
        ax.plot(goal.xs, goal.ys)
        fig.add_axes(ax)

        # The state
        xs = np.linspace(0, 1, 20)  # TODO: Change these values later, maybe more points
        f = lambdify(x, state.expr, 'numpy')
        ys = f(xs)
        ax.plot(xs, ys)
        fig.add_axes(ax)

        ax.legend(['goal', state.expr])


@domain_factory.register_parser("symbolic_regression")
class SymbolicParser(Parser):
    def parse(self, args_str: str) -> dict[str, Any]:
        return {"random_walk_length": int(args_str)}

    def help(self) -> str:
        return "An integer number of start states to generate through a random walk.'"


# For NN, may need to implement a different structure than Grid/Cube
# For example, our inputs here are flat. Maybe Transformer
# grid_heur class

# SymPy Notes
# evaluate expressions at a point by substituting a constant for a variable:
#   >> expr.subs(x, 0) to evaluate f(x=0)     # subs returns a new expression
#   >> expr.subs([(x, 2), (y, 3)])            # multiple variables

# evaluate a numerical expression like sqrt(2)
#   >> expr = sqrt(2)
#   >> expr.evalf()
