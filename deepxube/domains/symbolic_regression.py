from typing import List, Optional, Tuple

from deepxube.base.domain import State, Action, Goal, ActsEnumFixed, StartGoalWalkable, StringToAct, StateGoalVizable
from deepxube.factories.domain_factory import domain_factory

from sympy import Expr, simplify, Integer, symbols


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
# represents a set of states
class SymbolicGoal(Goal):
    def __init__(self, expression: Expr):
        self.f = expression
        # implemented incorrectly
        # goal here would be defined by the data that I want to do symbolic regression on


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
        # random walk from 0 - 10, sampling start states
        # symbolic_regression.10; need to implement parser class for this see GridParser for example

    def sample_start_states(self, num_states: int) -> List[SymbolicState]:
        # only start from empty function (0) for now
        return num_states * [SymbolicState(Expr(Integer(0)))]
        # want variety fo start/goal pairs
        # use all the intermediate states from walk (search)
    # sample start state, random walk, pick goal state
    #

    def sample_goal_from_state(self, states_start: Optional[List[SymbolicState]], states_goal: List[SymbolicState]) -> List[SymbolicGoal]:
        # Implemented like Grid. I'm not sure why this is necessary since it just copies States into Goals?
        return [SymbolicGoal(state_goal.f) for state_goal in states_goal]
        # one input variable
        # sample inputs, add noise (assume no noise for now); evaluate the state at some values (assume 0 - 1 for input for now)
        # fixed number of data points for simplicity
        # e.g. 20 x's --> 20 y's for set of next data points

        # simpler: assume 20 evenly spaced data points
        # NN takes in the 20 y points, maybe normalize


    # pickle file has the start states and goals for examples

    def get_actions_fixed(self) -> List[SymbolicAction]:
        return self.actions_fixed.copy()

    def next_state(self, states: List[SymbolicState], actions: List[SymbolicAction]) -> Tuple[List[SymbolicState], List[float]]:
        states_next: List[SymbolicState] = []

        # Should we simplify the expressions every time? Exploit this later; for now do not simplify

        for state, action in zip(states, actions):
            # temporary: just increase exponent
            if action.action == 0:
                if state.f == 0:
                    states_next.append(
                        SymbolicState(state.f + symbols('x'))
                    )
                else:
                    states_next.append(
                        SymbolicState(state.f * symbols('x'))
                    )
        # just using cost of 1 for now; this is fine, shortest path to the good expression should be simplest
        return states_next, [1.0] * len(states_next)

    def is_solved(self, states: List[SymbolicState], goals: List[SymbolicGoal]) -> List[bool]:
        # This assumes we know which function we're trying to get to, but shouldn't this evaluate the function
        # and see if it equals the sample values (i.e. the numbers)
        return [simplify(state.f - goal.f) == 0 for state, goal in zip(states, goals)]
        # this is wrong
        # should be: our state function evaluates to the data points within some tolerance (Hyperparameter)

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



# SymPy Notes
# evaluate expressions at a point by substituting a constant for a variable:
#   >> expr.subs(x, 0) to evaluate f(x=0)     # subs returns a new expression
#   >> expr.subs([(x, 2), (y, 3)])            # multiple variables

# evaluate a numerical expression like sqrt(2)
#   >> expr = sqrt(2)
#   >> expr.evalf()




