import math
from collections import namedtuple
from typing import List, Tuple

import torch
import torch.jit as jit
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter

def reverse(lst: List[Tensor]) -> List[Tensor]:
    return lst[::-1]

class SelectionLstmCell(jit.ScriptModule): # type: ignore
    """
    LSTM implementation based on [1] that adds a selection mechanism in front of the input and hidden
    neurons, with learnable parameters to 'learn structured sparsity'

    [1] L. Wen, X. Zhang, H. Bai, and Z. Xu, “Structured pruning of recurrent neural networks through neuron
    selection,” Neural Networks, vol. 123, pp. 134-141, Mar. 2020. DOI: 10.1016/j.neunet.2019.11.018
    """

    def __init__(self, input_size, hidden_size):
        super(SelectionLstmCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = Parameter(torch.empty(4 * self.hidden_size, self.input_size))
        self.U = Parameter(torch.empty(4 * self.hidden_size, self.hidden_size))
        self.b = Parameter(torch.empty(4 * self.hidden_size, dtype=torch.float16))
        # self.log_alpha_z = Parameter(torch.empty(self.input_size, dtype=torch.float16))
        self.log_alpha_s = Parameter(torch.empty(self.hidden_size, dtype=torch.float16))

        # Hyper-parameters
        self.beta = SelectionLstm.BETA
        self.gamma = SelectionLstm.GAMMA
        self.zeta = SelectionLstm.ZETA

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, weight in self.named_parameters():
            if name == 'z' or name == 's':
                weight.data.fill_(1)
            elif name == 'log_alpha_z' or name == 'log_alpha_s':
                # "For log α, it is updated by back-propagation of the network and initialized by samples from N(1, 0.1)"
                weight.data.normal_(mean=1, std=0.1)
            else:
                weight.data.uniform_(-stdv, +stdv)

    @jit.script_method # type: ignore
    def _smooth_gate(self, log_alpha: Tensor) -> Tensor:
        u = torch.rand(log_alpha.shape, device=log_alpha.device) # pyright: ignore
        τ_hat = torch.sigmoid((u.log() - (-u).log1p() + log_alpha) / self.beta)
        τ = τ_hat * (self.zeta - self.gamma) + self.gamma
        return τ.clamp(0, 1)

    @jit.script_method # type: ignore
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state

        # z = self._smooth_gate(self.log_alpha_z)
        s = self._smooth_gate(self.log_alpha_s)

        # W_hat = self.W * (z.unsqueeze(1) @ s.unsqueeze(0)).repeat(4, 1)
        W_hat = self.W
        U_hat = self.U * (s.unsqueeze(1) @ s.unsqueeze(0)).repeat(4, 1)

        gates = (
            torch.mm(input, W_hat.t())
            + torch.mm(hx, U_hat.t())
            + self.b
        )
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)

class SelectionLstm(jit.ScriptModule): # type: ignore
    BETA = 2/3 # "Try lambda = 2/3" (https://vitalab.github.io/article/2018/11/29/concrete.html)
    GAMMA = -0.1 # https://arxiv.org/pdf/1811.09332
    ZETA = 1.1

    def __init__(self, input_size, hidden_size, reverse=False):
        super(SelectionLstm, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reverse = reverse

        self.lstm = SelectionLstmCell(input_size, hidden_size)

    @jit.script_method # type: ignore
    def forward(self, input: Tensor) -> Tensor:
        # x: [sequence len, batch size, input size]
        # e.g. [400       , 64        , 384       ]

        h0 = torch.zeros(input.size(1), 384, device=input.device)
        c0 = torch.zeros(input.size(1), 384, device=input.device)
        state = (h0, c0)

        inputs = reverse(input.unbind(0)) if self.reverse else input.unbind(0)
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.lstm(inputs[i], state)
            outputs += [out]

        return torch.stack(reverse(outputs) if self.reverse else outputs)#, state
