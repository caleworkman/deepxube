from typing import List, Optional, Any

from deepxube.base.factory import Parser
from deepxube.base.domain import State, Action, Goal, ActsEnumFixed, StartGoalWalkable, StringToAct, StateGoalVizable
from deepxube.factories.domain_factory import domain_factory

import numpy as np
from sympy import *
from sympy.abc import x

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


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
    ActsEnumFixed[SymbolicState, SymbolicAction, SymbolicGoal], # ActsEnum; can limit to things like doubling or halving
    StartGoalWalkable[SymbolicState, SymbolicAction, SymbolicGoal],
    StateGoalVizable[SymbolicState, SymbolicAction, SymbolicGoal],
    StringToAct[SymbolicState, SymbolicAction, SymbolicGoal]
):

    def __init__(self, num_start_states: int):
        # Test: python -m deepxube viz --domain symbolic_regression.1
        super().__init__()
        self.actions_fixed: List[SymbolicAction] = [SymbolicAction(n) for n in [0]]
        self.num_start_states = num_start_states

    def sample_start_states(self, num_states: int) -> List[SymbolicState]:
        return num_states * [SymbolicState(x)]
        # TODO: Variety in start/goal pairs.
        # 1. Sample start state from empty function (generate goal pairs by evaluating the function)
        # 2. Random walk - pick an action and do it
        # 3. Sample this next state
        # 4. Repeat num_states times

    def sample_goal_from_state(self, states_start: Optional[list[SymbolicState]], states_goal: list[SymbolicState]) -> list[SymbolicGoal]:
        noise = 0  # TODO: Introduce noise later
        xs = np.linspace(0.0001, 1, 20)  # TODO: Change these values later, maybe more points
        xs = np.add(xs, noise)

        goals = []
        for state in states_goal:
            f = lambdify(x, state.expr, 'numpy')
            ys = f(xs)
            goals.append(SymbolicGoal(xs=xs, ys=ys, tolerance=0))
        return goals

    def get_actions_fixed(self) -> List[SymbolicAction]:
        return self.actions_fixed.copy()

    def next_state(self, states: list[SymbolicState], actions: list[SymbolicAction]) -> tuple[list[SymbolicState], list[float]]:
        states_next: List[SymbolicState] = []

        # Should we simplify the expressions every time? Exploit this later; for now do not simplify

        for state, action in zip(states, actions):
            # temporary: just increase exponent
            if action.action == 0:
                if state.expr == 0:
                    states_next.append(
                        SymbolicState(state.expr + symbols('x', real=True))
                    )
                else:
                    states_next.append(
                        SymbolicState(state.expr * symbols('x', real=True))
                    )
        return states_next, [1.0] * len(states_next)

    def is_solved(self, states: list[SymbolicState], goals: list[SymbolicGoal]) -> list[bool]:
        # Evaluate each state at the x's in goals, and see if the y's match within some tolerance (later).
        solved = []
        tolerance = 0

        for state, goal in zip(states, goals):
            f = lambdify(x, state.expr, 'numpy')
            y_values = f(goal.xs)
            within_tolerance = np.less(np.abs(goal.ys - y_values), tolerance)
            solved.append(np.all(within_tolerance))
        return solved

    def string_to_action(self, act_str: str) -> Optional[SymbolicAction]:
        if act_str in {"0"}:
            return SymbolicAction(int(act_str))
        else:
            return None

    def string_to_action_help(self) -> str:
        return "0 to increase the exponent"

    def visualize_state_goal(self, state: SymbolicState, goal: SymbolicGoal, fig: Figure) -> None:
        # can create a figure with the data points, and the function (state) overlaid
        ax = plt.axes()

        # The goal
        ax.plot(goal.xs, goal.ys)
        fig.add_axes(ax)

        # The state
        xs = np.linspace(0.01, 1, 20)  # TODO: Change these values later, maybe more points
        f = lambdify(x, state.expr, 'numpy')
        ys = f(xs)
        ax.plot(xs, ys)
        fig.add_axes(ax)

        ax.legend(['goal', 'state'])


@domain_factory.register_parser("symbolic_regression")
class SymbolicParser(Parser):
    def parse(self, args_str: str) -> dict[str, Any]:
        return {"num_start_states": int(args_str)}

    def help(self) -> str:
        return "An integer number of start states to generate through a random walk.'"

# SymPy Notes
# evaluate expressions at a point by substituting a constant for a variable:
#   >> expr.subs(x, 0) to evaluate f(x=0)     # subs returns a new expression
#   >> expr.subs([(x, 2), (y, 3)])            # multiple variables

# evaluate a numerical expression like sqrt(2)
#   >> expr = sqrt(2)
#   >> expr.evalf()
