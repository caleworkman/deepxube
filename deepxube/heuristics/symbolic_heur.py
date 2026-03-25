from typing import Type, List

from torch import nn, Tensor

from deepxube.base.factory import Parser
from deepxube.base.heuristic import HeurNNet, In
from deepxube.nnet.pytorch_models import Conv2dModel, FullyConnectedModel
from deepxube.factories.heuristic_factory import heuristic_factory

from deepxube.domains.symbolic_regression import SymbolicRegressionNNetInput


@heuristic_factory.register_class("symbolic_regression_net")
class SymbolicRegressionNet(HeurNNet[SymbolicRegressionNNetInput]):

    @staticmethod
    def nnet_input_type() -> Type[SymbolicRegressionNNetInput]:
        return SymbolicRegressionNNetInput

    def __init__(
            self,
            nnet_input: SymbolicRegressionNNetInput,
            out_dim: int,
            q_fix: bool,
            chan_size: int = 8,
            fc_size: int = 100
    ):
        # copied from Grid
        self.heur: nn.Module = nn.Sequential(
            Conv2dModel(2, [chan_size, chan_size], [3, 3], [1, 1], ["RELU", "RELU"], batch_norms=[True, True]),
            nn.Flatten(),
            FullyConnectedModel(grid_dim * grid_dim * chan_size, [fc_size], ["RELU"], batch_norms=[True]),
            nn.Linear(fc_size, self.out_dim)
        )

    def _forward(self, inputs: List[Tensor]) -> Tensor:
        pass


# no arguments, means no need for parser class
# the parser just modifies hyperparameters, explore this later

# transformer to tokenize the expression
# implement to_np functions
# papers on chemical reactions