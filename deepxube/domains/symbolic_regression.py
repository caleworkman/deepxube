from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from matplotlib.figure import Figure

from deepxube.base.factory import Parser
from deepxube.base.domain import State, Action, Goal, ActsEnumFixed, StartGoalWalkable, StringToAct, StateGoalVizable
from deepxube.factories.domain_factory import domain_factory

from sympy import Expr, simplify, Integer, symbols


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
    def __init__(self, points: List[Tuple[float, float]], tolerance: float):
        """An expression should evaluate to the pairs of values (x, y), eventually implement tolerance."""
        self.points = points
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
        super().__init__()
        self.actions_fixed: List[SymbolicAction] = [SymbolicAction(x) for x in [0]]
        self.num_start_states = num_start_states
        # TODO: Calling "symbolic_regression.10" should set num_start_states = 10 if Parser is implemented

    def sample_start_states(self, num_states: int) -> List[SymbolicState]:
        return num_states * [SymbolicState(Expr(Integer(0)))]
        # TODO: Variety in start/goal pairs.
        # 1. Sample start state from empty function (generate goal pairs by evaluating the function)
        # 2. Random walk - pick an action and do it
        # 3. Sample this next state
        # 4. Repeat num_states times

    def sample_goal_from_state(self, states_start: Optional[List[SymbolicState]], states_goal: List[SymbolicState]) -> List[SymbolicGoal]:
        noise = 0   # TODO: Introduce noise later
        xs = np.linspace(0, 1, 20)  # TODO: Change these values later, maybe more points
        xs = [x + noise for x in xs]

        goals = []
        for state in states_goal:
            points = []
            for x_value in xs:
                expr = state.expr.subs(symbols('x'), x_value)
                points.append(
                    (x_value, expr.evalf())
                )
            goals.append(SymbolicGoal(points=points))
        return goals

    def get_actions_fixed(self) -> List[SymbolicAction]:
        return self.actions_fixed.copy()

    def next_state(self, states: List[SymbolicState], actions: List[SymbolicAction]) -> Tuple[List[SymbolicState], List[float]]:
        states_next: List[SymbolicState] = []

        # Should we simplify the expressions every time? Exploit this later; for now do not simplify

        for state, action in zip(states, actions):
            # temporary: just increase exponent
            if action.action == 0:
                if state.expr == 0:
                    states_next.append(
                        SymbolicState(state.expr + symbols('x'))
                    )
                else:
                    states_next.append(
                        SymbolicState(state.expr * symbols('x'))
                    )
        return states_next, [1.0] * len(states_next)

    def is_solved(self, states: List[SymbolicState], goals: List[SymbolicGoal]) -> List[bool]:
        # Evaluate each state at the x's in goals, and see if the y's match within some tolerance (later).
        solved = []
        tolerance = 0
        for state in states:
            for x_goal, y_goal in goals.points:
                expr = state.expr.subs(symbols('x'), x_goal)
                within_tolerance = np.abs(y_goal - expr.evalf()) <= tolerance
                if within_tolerance:
                    solved.append(False)
                else:
                    continue
            solved.append(True)
        return solved

    def string_to_action(self, act_str: str) -> Optional[SymbolicAction]:
        if act_str in {"0"}:
            return SymbolicAction(int(act_str))
        else:
            return None

    def string_to_action_help(self) -> str:
        return "0 to increase the exponent"

    def visualize_state_goal(self, state: SymbolicState, goal: SymbolicGoal, fig: Figure) -> None:
        pass
        # can create a figure with the data points, and the function (state) overlaid


@domain_factory.register_parser("symbolic_regression")
class SymbolicParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
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




