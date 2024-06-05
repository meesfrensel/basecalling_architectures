import math
from typing import List, Tuple

import torch
import torch.jit as jit
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
        self.b = Parameter(torch.empty(4 * self.hidden_size))
        # self.log_alpha_z = Parameter(torch.empty(self.input_size))
        self.log_alpha_s = Parameter(torch.empty(self.hidden_size))

        # Hyper-parameters
        self.beta = SelectionLstm.BETA
        self.gamma = SelectionLstm.GAMMA
        self.zeta = SelectionLstm.ZETA

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, weight in self.named_parameters():
            if name == 'log_alpha_s':
                # "For log α, it is updated by back-propagation of the network and initialized by samples from N(1, 0.1)"
                weight.data.normal_(mean=1, std=0.1)
            else:
                weight.data.uniform_(-stdv, +stdv)

    @jit.script_method # type: ignore
    def _smooth_gate(self, log_alpha: Tensor) -> Tensor:
        u = torch.rand_like(log_alpha)
        τ_hat = torch.sigmoid((u.log() - (-u).log1p() + log_alpha) / self.beta)
        τ = τ_hat * (self.zeta - self.gamma) + self.gamma
        return τ.clamp(0, 1)

    @jit.script_method # type: ignore
    def _estimate_final_gate(self, log_alpha: Tensor) -> Tensor:
        return torch.clamp(torch.sigmoid(log_alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    @jit.script_method # type: ignore
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state

        if self.training:
            s = self._smooth_gate(self.log_alpha_s)
        else:
            s = self._estimate_final_gate(self.log_alpha_s)

        W_hat = self.W * s.unsqueeze(0)
        # W_hat = self.W
        U_hat = self.U * (s.unsqueeze(1) @ s.unsqueeze(0)).repeat(4, 1)
        # U_hat = self.U

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

class SelectionLstm(torch.nn.Module):
    BETA = 2/3 # "Try lambda = 2/3" (https://vitalab.github.io/article/2018/11/29/concrete.html)
    GAMMA = -0.1 # https://arxiv.org/pdf/1811.09332
    ZETA = 1.1

    def __init__(self, input_size, hidden_size, reverse=False):
        super(SelectionLstm, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reverse = reverse

        self.lstm = SelectionLstmCell(input_size, hidden_size)

        self.inference_lstm = torch.nn.LSTM(input_size, hidden_size)
        # self.register_load_state_dict_post_hook(change_lstm_cell)
        self.must_update_inference_lstm = False
        self.mask = torch.ones(hidden_size, dtype=torch.bool, device='cuda')

    def forward(self, input: Tensor) -> Tensor:
        # x: [sequence len, batch size, input size]
        # e.g. [400       , 64        , 384       ]
        batch_size = input.shape[1]

        # if not self.training: # inference
        #     if self.must_update_inference_lstm:
        #         change_lstm_cell(self)

        #     output, _ = self.inference_lstm(input.flip(0) if self.reverse else input)
        #     tmp = torch.zeros(input.shape[0], batch_size, self.hidden_size, device=input.device) # original hidden size, not the pruned/masked hidden size
        #     output = tmp.masked_scatter_(self.mask.unsqueeze(0).unsqueeze(1), output)
        #     assert output.shape == torch.Size([input.shape[0], input.shape[1], self.hidden_size])
        #     return output

        self.must_update_inference_lstm = True

        hidden_size = input.shape[2] if self.training else int(self.mask.count_nonzero().item())
        h0 = torch.zeros(batch_size, hidden_size, device=input.device)
        c0 = torch.zeros(batch_size, hidden_size, device=input.device)
        state = (h0, c0)

        inputs = reverse(input.unbind(0)) if self.reverse else input.unbind(0)
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.lstm(inputs[i], state)
            outputs += [out]

        return torch.stack(reverse(outputs) if self.reverse else outputs)#, state

def change_lstm_cell(module: SelectionLstm, _unmappable=None):
    THRESHOLD = 0.3
    s = module.lstm._estimate_final_gate(module.lstm.log_alpha_s)
    s[s <= THRESHOLD] = 0
    s[s > THRESHOLD] = 1

    # Shrink weight matrices by removing all zero columns & rows
    num_nonzero = int(s.count_nonzero().item())

    mask_W = s.repeat(s.shape[0] * 4, 1).type(torch.bool)
    mask_U = (s.unsqueeze(1) @ s.unsqueeze(0)).repeat(4, 1).type(torch.bool)
    # TODO: check impact of also masking bias; these terms might be useful for final accuracy
    bias = module.lstm.b.masked_select(s.type(torch.bool).repeat(4)).reshape(num_nonzero * 4)

    new_dict = {
        'weight_ih_l0': module.lstm.W.masked_select(mask_W).reshape(num_nonzero * 4, module.lstm.W.shape[1]),
        'weight_hh_l0': module.lstm.U.masked_select(mask_U).reshape(num_nonzero * 4, num_nonzero),
        'bias_ih_l0': bias,
        'bias_hh_l0': torch.zeros_like(bias),
    }
    module.inference_lstm = torch.nn.LSTM(module.input_size, num_nonzero)
    module.inference_lstm.load_state_dict(new_dict)
    module.mask = s.type(torch.bool).to('cuda')
