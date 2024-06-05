"""Implementation of the Bonito model with linear recurrence

Based on: 
https://github.com/nanoporetech/bonito
"""

import math
import os
import sys
import torch
from torch import nn
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from classes import BaseModelImpl
from selection_lstm import SelectionLstm

class OwnModel(BaseModelImpl):
    """Bonito Model adapted with linear recurrence
    """
    def __init__(self, convolution = None, encoder = None, decoder = None, reverse = True, load_default = False, *args, **kwargs):
        super(OwnModel, self).__init__(*args, **kwargs)
        """
        Args:
            convolution (nn.Module): module with: in [batch, channel, len]; out [batch, channel, len]
            encoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
            decoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
            reverse (bool): if the first rnn layer starts with reverse 
        """
    
        self.convolution = convolution
        self.encoder = encoder
        self.decoder = decoder
        self.reverse = reverse
        
        if load_default:
            self.load_default_configuration()

    def forward(self, x):
        """Forward pass of a batch
        
        Args:
            x (tensor) : [batch size, channels (1), seq len]
        """
        
        x = self.convolution(x)
        x = x.permute(2, 0, 1) # [seq len, batch size, channels (=hidden size)]
        # start = time.time()
        x = self.encoder(x)
        # end = time.time()
        # if not self.training:
        #     print("elapsed: {:.3f}".format(end - start))
        x = self.decoder(x)
        return x

    def build_cnn(self):

        cnn = nn.Sequential(
            nn.Conv1d(
                in_channels = 1, 
                out_channels = 4, 
                kernel_size = 5, 
                stride= 1, 
                padding=5//2, 
                bias=True),
            nn.SiLU(),
            nn.Conv1d(
                in_channels = 4, 
                out_channels = 16, 
                kernel_size = 5, 
                stride= 1, 
                padding=5//2, 
                bias=True),
            nn.SiLU(),
            nn.Conv1d(
                in_channels = 16, 
                out_channels = 384, 
                kernel_size = 19, 
                stride= 5, 
                padding=19//2, 
                bias=True),
            nn.SiLU()
        )
        return cnn

    def build_encoder(self, input_size, reverse):

        if reverse:
            encoder = nn.Sequential(SelectionLstm(input_size, 384, reverse = True),
                                    SelectionLstm(384, 384, reverse = False),
                                    SelectionLstm(384, 384, reverse = True),
                                    SelectionLstm(384, 384, reverse = False),
                                    SelectionLstm(384, 384, reverse = True),
                                    )
        else:
            encoder = nn.Sequential(SelectionLstm(input_size, 384, reverse = False),
                                    SelectionLstm(384, 384, reverse = True),
                                    SelectionLstm(384, 384, reverse = False),
                                    SelectionLstm(384, 384, reverse = True),
                                    SelectionLstm(384, 384, reverse = False),
                                    )
        return encoder    

    def get_defaults(self):
        defaults = {
            'cnn_output_size': 384, 
            'cnn_output_activation': 'silu',
            'encoder_input_size': 384,
            'encoder_output_size': 384,
            'cnn_stride': 5,
        }
        return defaults
        
    def load_default_configuration(self):
        """Sets the default configuration for one or more
        modules of the network
        """

        self.convolution = self.build_cnn()
        self.cnn_stride = self.get_defaults()['cnn_stride']
        self.encoder = self.build_encoder(input_size = 384, reverse = True)
        self.decoder = self.build_decoder(encoder_output_size = 384, decoder_type = 'crf')
        self.decoder_type = 'crf'
    
    def _nonzero_cdf(self, log_alpha):
        return torch.sigmoid(torch.add(log_alpha, -SelectionLstm.BETA * math.log(-SelectionLstm.GAMMA / SelectionLstm.ZETA)))

    def train_step(self, batch):
        """Copied from classes.py>BaseModel, and added loss for nonzero z or s elements.

        Train a step with a batch of data.
        
        Args:
            batch (dict): dict with keys 'x' (batch, len) 
                                         'y' (batch, len)
        """
        
        self.train()
        x = batch['x'].to(self.device)
        x = x.unsqueeze(1) # add channels dimension
        y = batch['y'].to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            p = self.forward(x) # forward through the network
            loss, losses = self.calculate_loss(y, p)

        lambda_ = 0.0000001 # 0.08/N (if N is correctly assumed to be 384) is about 0.0002

        for name, param in self.named_parameters():
            if 'log_alpha_s' in name:
                s_expectation = self._nonzero_cdf(param)

                # Input-to-hidden. We pretend z is all ones. TODO: get 384 from input size
                loss += (lambda_ * 384 * s_expectation).sum()

                # Hidden-to-hidden, diagonal only
                loss += (lambda_ * s_expectation).sum()

                # Hidden-to-hidden, except diagonal
                loss += (lambda_ * torch.einsum('i,j->ij', s_expectation, s_expectation)).fill_diagonal_(0).sum()

        self.optimize(loss)
        losses['loss.global'] = loss.item()
        
        return losses, p
