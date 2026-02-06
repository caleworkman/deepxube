from typing import List, Optional

from deepxube.base.domain import State, Action, Goal, ActsEnumFixed, StartGoalWalkable, StringToAct
from deepxube.factories.domain_factory import domain_factory

from sympy import *


class SymbolicState(State):
    def __init__(self, expression: Expr):
        self.f = expression

    def __hash__(self) -> int:
        return hash(self.f)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SymbolicState):
            # SymPy tests for equality by subtracting one function from another and testing for 0
            # https://docs.sympy.org/latest/tutorials/intro-tutorial/gotchas.html
            return simplify(self.f - other.f) == 0
        return NotImplemented


# TODO: Question. Why does Goal need to be its own class? Shouldn't it just be a state?
class SymbolicGoal(Goal):
    def __init__(self, expression: Expr):
        self.f = expression


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
    StringToAct[SymbolicState, SymbolicAction, SymbolicGoal]
):

    def __init__(self):
        super().__init__()
        self.actions_fixed: List[SymbolicAction] = [SymbolicAction(x) for x in [0]]

    def sample_start_states(self, num_states: int) -> List[SymbolicState]:
        """ A method for generating start states. Should try to make this generate states that are as diverse as
        possible so that the trained heuristic function generalizes well.

        :param num_states: Number of states to get
        :return: Generated states
        """
        return num_states * [0]

    def sample_goal_from_state(self, states_start: Optional[List[SymbolicState]], states_goal: List[SymbolicState]) -> List[SymbolicGoal]:
        """ Given a state, sample a goal that represents a set of goal states of which the given state is a member.

        :param states_start: Optional list of start states. Can be used to sample goals that are difficult to achieve
            from the given start state.
        :param states_goal: List of states from which goals will be sampled.
        :return: Goals
        """
        # Implemented like Grid. I'm not sure why this is necessary since it just copies States into Goals?
        return [SymbolicGoal(state_goal.f) for state_goal in states_goal]

    def get_actions_fixed(self) -> List[SymbolicAction]:
        return self.actions_fixed.copy()

    def next_state(self, states: List[SymbolicState], actions: List[SymbolicAction]) -> Tuple[List[SymbolicState], List[float]]:
        """ Get the next state and transition cost given the current state and action

        :param states: List of states
        :param actions: List of actions to take
        :return: Next states, transition costs
        """
        states_next: List[SymbolicState] = []

        # Should we simplify the expressions every time?

        for state, action in zip(states, actions):
            # temporary just increase exponent
            if action.action == 0:
                if state.f == 0:
                    states_next.append(
                        SymbolicState(state.f + symbols('x'))
                    )
                else:
                    # increment the exponent
                    states_next.append(
                        SymbolicState(state.f * symbols('x'))
                    )
        # just using cost of 1 for now
        return states_next, [1.0] * len(states_next)

    def is_solved(self, states: List[SymbolicState], goals: List[SymbolicGoal]) -> List[bool]:
        # This assumes we know which function we're trying to get to, but shouldn't this evaluate the function
        # and see if it equals the sample values (i.e. the numbers)
        return [simplify(state.f - goal.f) == 0 for state, goal in zip(states, goals)]


# SymPy Notes
# evaluate expressions at a point by substituting a constant for a variable:
#   >> expr.subs(x, 0) to evaluate f(x=0)     # this returns a new expression
#   >> expr.subs([(x, 2), (y, 3)])            # multiple variables

# evaluate a numerical expression like sqrt(2)
#   >> expr = sqrt(2)
#   >> expr.evalf()




