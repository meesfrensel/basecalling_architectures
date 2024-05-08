"""Implementation of the Bonito model with linear recurrence

Based on: 
https://github.com/nanoporetech/bonito
"""

import math
import os
import sys
import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from classes import BaseModelImpl
from layers.bonito import BonitoLSTM
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
        x = self.encoder(x)
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
                                    # SelectionLstm(384, 384, reverse = False),
                                    # SelectionLstm(384, 384, reverse = True),
                                    )
        else:
            encoder = nn.Sequential(SelectionLstm(input_size, 384, reverse = False),
                                    SelectionLstm(384, 384, reverse = True),
                                    SelectionLstm(384, 384, reverse = False),
                                    # SelectionLstm(384, 384, reverse = True),
                                    # SelectionLstm(384, 384, reverse = False),
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
        return torch.sigmoid(torch.add(log_alpha, 0.67 * math.log(-(1.1) / -0.1)))

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

        lambda_1 = 0.000005 # 0.08/N (if N is correctly assumed to be 384) is about 0.0002
        lambda_2 = 0.000005

        log_alpha_z = None
        log_alpha_s = None
        for name, param in self.named_parameters():
            if 'log_alpha_z' in name:
                log_alpha_z = param.clamp(0, 1)

            if 'log_alpha_s' in name:
                log_alpha_s = param.clamp(0, 1)
                loss += lambda_2 * self._nonzero_cdf(log_alpha_s).sum().item()
                
                einsum = torch.einsum('i,j->ij',
                                      self._nonzero_cdf(log_alpha_s),
                                      self._nonzero_cdf(log_alpha_s)
                                      ).fill_diagonal_(0).sum()
                loss += lambda_2 * (einsum.item() if einsum.isfinite() else 65000)

            if 'log_alpha' in name and log_alpha_z != None and log_alpha_s != None:
                einsum = torch.einsum('i,j->ij',
                                      self._nonzero_cdf(log_alpha_z),
                                      self._nonzero_cdf(log_alpha_s)
                                      ).sum()
                loss += lambda_1 * (einsum.item() if einsum.isfinite() else 65000)

                log_alpha_z = None
                log_alpha_s = None

        self.optimize(loss)
        losses['loss.global'] = loss.item()
        
        return losses, p
