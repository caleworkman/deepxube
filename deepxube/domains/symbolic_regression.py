from typing import List, Optional, Any

from numpy._typing import NDArray

from deepxube.base.factory import Parser
from deepxube.base.domain import State, Action, Goal, ActsEnum, StartGoalWalkable, StringToAct, StateGoalVizable, A
from deepxube.base.nnet_input import StateGoalIn, G

from deepxube.factories.domain_factory import domain_factory
from deepxube.factories.nnet_input_factory import register_nnet_input

import numpy as np
from sympy import *
from sympy.abc import x

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from enum import IntEnum, auto


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


class SymbolicActionEnum(IntEnum):
    ADD_1 = auto()
    ADD_X = auto()

    SUBTRACT_1 = auto()
    SUBTRACT_X = auto()

    MULTIPLY_BY_2 = auto()
    MULTIPLY_BY_X = auto()

    DIVIDE_BY_2 = auto()
    DIVIDE_BY_X = auto()

    MULTIPLY_BY_NEG_1 = auto()


class SymbolicAction(Action):
    def __init__(self, term: int, action: SymbolicActionEnum, value: float | Symbol = None):
        """An action

        :param int term: The term in the expression to modify; -1 modifies the entire expression.
        :param SymbolicActionEnum action: The action to take
        :param float | symbol value: The value to add/multiply/etc to the expression/term
        """
        self.term = term
        self.action = action
        # self.value = value # don't do this yet

    def __hash__(self) -> int:
        # TODO: how to hash the term? There will never be 1000 possible actions
        return 1000*self.term + self.action

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SymbolicAction):
            return self.action == other.action and self.term == other.term
        return NotImplemented

    def __repr__(self) -> str:
        return f"(Term Index: {self.term}, Action: {self.action})"


@domain_factory.register_class('symbolic_regression')
class SymbolicRegression(
    ActsEnum[SymbolicState, SymbolicAction, SymbolicGoal],
    StartGoalWalkable[SymbolicState, SymbolicAction, SymbolicGoal],
    StateGoalVizable[SymbolicState, SymbolicAction, SymbolicGoal],
    StringToAct[SymbolicState, SymbolicAction, SymbolicGoal]
):

    def __init__(self, random_walk_length: int):
        # Test: python -m deepxube viz --domain symbolic_regression.1 --steps 5
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
            print('Goal: ', state.expr)
            f = lambdify(x, state.expr, 'numpy')
            ys = f(xs)
            goals.append(SymbolicGoal(xs=xs, ys=ys, tolerance=0))
        return goals

    # def get_actions_fixed(self) -> List[SymbolicAction]:
    #     return [SymbolicAction(term=1, action=n) for n in range(0, len(SymbolicActionEnum))]

    def get_state_actions(self, states: list[SymbolicState]) -> list[list[SymbolicAction]]:
        list_of_list_of_actions = []

        for state in states:
            num_terms = len(state.expr.args)

            if num_terms < 2:
                state_actions = [SymbolicAction(-1, a) for a in SymbolicActionEnum]
            else:
                term_idxs = [-1] + list(range(num_terms))
                state_actions = [SymbolicAction(t, a) for a in SymbolicActionEnum for t in term_idxs]

            list_of_list_of_actions.append(state_actions)

        return list_of_list_of_actions

    def next_state(self, states: list[SymbolicState], actions: list[SymbolicAction]) -> tuple[list[SymbolicState], list[float]]:
        states_next: List[SymbolicState] = []

        for state, action in zip(states, actions):
            terms = list(state.expr.args)
            if action.term < 0 or len(terms) < 2:
                # apply to entire expression
                new_expr = self._apply_action(state.expr, action.action)
            else:
                terms[action.term] = self._apply_action(terms[action.term], action.action)
                new_expr = state.expr.func(*terms)

            states_next.append(SymbolicState(simplify(new_expr)))
        return states_next, [1.0] * len(states_next)

    @staticmethod
    def _apply_action(term, action: SymbolicActionEnum):
        """Apply a sympy manipulation to a term of an expression."""
        if action == SymbolicActionEnum.ADD_1:
            return term + 1
        elif action == SymbolicActionEnum.ADD_X:
            return term + x

        elif action == SymbolicActionEnum.SUBTRACT_1:
            return term - 1
        elif action == SymbolicActionEnum.SUBTRACT_X:
            return term - x

        elif action == SymbolicActionEnum.MULTIPLY_BY_2:
            return term * 2
        elif action == SymbolicActionEnum.MULTIPLY_BY_X:
            return term * x

        elif action == SymbolicActionEnum.DIVIDE_BY_2:
            return term / 2
        elif action == SymbolicActionEnum.DIVIDE_BY_X:
            return term / x

        elif action == SymbolicActionEnum.MULTIPLY_BY_NEG_1:
            return -1 * term

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
        # TODO: make this better, check format and number of valid terms
        terms = act_str.split(',')
        return SymbolicAction(
            term=int(terms[0]),
            action=int(terms[1]),
            # value=symbols(terms[2]) if isinstance(terms[2], str) else float(terms[2])
        )

    def string_to_action_help(self) -> str:
        return '\n'.join([f'{a.value}: {a.name}' for a in SymbolicActionEnum])

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


@register_nnet_input("symbolic_regression", "symbolic_regression_nnet_input")
class SymbolicRegressionNNetInput(StateGoalIn[SymbolicRegression, SymbolicState, SymbolicGoal]):
    def get_input_info(self) -> Any:
        pass

    def to_np(self, states: List[SymbolicState], goals: List[SymbolicGoal]) -> List[NDArray]:
        # Final input something like
        # [
        #   [
        #       [token1_1, token1_2, ..., token_1n],    # fixed length, assume max expression length and mask (pad?) missing
        #       [token2_1, token2_2, ..., token2_n],
        #   ],
        #   [x1, x2, ..., x_n],    # this can probably be one dim because all the points are the same
        #   [
        #       [y1_1, y2, ..., y_n],
        #       [y2_1, y2, ..., y_n],
        #   ],
        #   [
        #       [g1_1, g1_2, ..., g1_n],
        #       [g2_1, g2_2, ..., g2_n]
        #   ]
        # ]

        # there are multiple token's, y's, and g's because it expects examples which it can distribute across GPUs, etc
        pass


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
