from typing import List

from deepxube.base.domain import State, Action, Goal, ActsEnumFixed, StartGoalWalkable, G
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
            # TODO I think we'll need to compare before and afters
            # i.e. using simplify(a - b) == 0 to check if an action is the same?
            return self.action == other.action
        return NotImplemented

    def __repr__(self) -> str:
        return f"{self.action}"


@domain_factory.register_class('symbolic regression')
class SymbolicRegression(
    ActsEnumFixed[SymbolicState, SymbolicAction, SymbolicGoal],
    StartGoalWalkable[SymbolicState, SymbolicAction, SymbolicGoal]
):

    # TODO: Why does Grid class not need to implement this abstractmethod?
    def sample_start_goal_pairs(self, num_steps_l: List[int], times: Optional[Times] = None) -> Tuple[List[S], List[G]]:
        """ Return start goal pairs with num_steps_l between start and goal

        :param num_steps_l: Number of steps to take between start and goal
        :param times: Times that can be used to profile code
        :return: List of start states and list of goals
        """
        pass

    def sample_state_action(self, states: List[SymbolicState]) -> List[SymbolicAction]:
        """ Get a random action that is applicable to the current state

        :param states: List of states
        :return: List of random actions applicable to given states
        """
        pass

    def next_state(self, states: List[SymbolicState], actions: List[SymbolicAction]) -> Tuple[List[SymbolicState], List[float]]:
        """ Get the next state and transition cost given the current state and action

        :param states: List of states
        :param actions: List of actions to take
        :return: Next states, transition costs
        """
        pass

    def is_solved(self, states: List[SymbolicState], goals: List[SymbolicGoal]) -> List[bool]:
        pass

