"""
Script for iteratively pruning and finetuning pretrained models
"""

import argparse
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.utils.prune
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))

from bonito.model import BonitoModel as Model # pyright: ignore[reportMissingImports]
#from own.model import OwnModel as Model # pyright: ignore[reportMissingImports]
from classes import BaseNanoporeDataset
from constants import NON_RECURRENT_DECODING_DICT, NON_RECURRENT_ENCODING_DICT
from schedulers import GradualWarmupScheduler
from sparse_rnn_core import CosineDecay, Masking, add_sparse_args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STEPS_PER_VALIDATION = 500
STEPS_PER_CHECKPOINT = 20000

def generate_log_df(losses_keys, metrics_keys):
    """Creates a data.frame to store the logging values
    """
    
    header = ['epoch', # epoch number
              'step',  # step number
              'time']  # time it took
    # add losses and metrics for train and validation
    for k in losses_keys:
        header.append(k + '.train')
        header.append(k + '.val')
    for k in metrics_keys:
        header.append(k + '.train')
        header.append(k + '.val')
    # whether a checkpoint was saved at this step
    header.append('lr')
    header.append('checkpoint')
    
    log_dict = dict()
    for k in header:
        log_dict[k] = [None]
    return pd.DataFrame(log_dict)

class ColumnarPruningMethod(torch.nn.utils.prune.BasePruningMethod):
    PRUNING_TYPE = 'structured'

    def __init__(self, amount, dim):
        assert amount >= 0.0
        self.amount = amount
        self.dim = dim
        assert dim == 0

    def compute_mask(self, t, default_mask):
        if self.amount == 0.0:
            return default_mask

        mask = default_mask.clone()
        tensr = t.detach()
        values, indices = torch.sort(torch.sum(torch.abs(tensr * mask), dim=self.dim)) # find smallest columns; that warrants pruning
        offset = 0
        for val in values:
            if val > 0.0:
                break
            offset += 1

        mask[:, indices[offset:int(offset + self.amount * len(indices - offset))]] = 0 # set mask to 0 for those columns
        return mask

def main(model, num_iterations, prune_config, epochs_per_iter):
    torch.set_printoptions(edgeitems=5, linewidth=200)

    for i in range(num_iterations):
        if model.mask == None: # otherwise mask pruning is done in train_step
            model = prune(model, prune_config)
        model = finetune(model, epochs_per_iter)
        model.save(os.path.join(checkpoints_dir, 'checkpoint_iter_' + str(i) + '.pt'))

def prune(model, prune_config):
    for layer, prune_amount in prune_config.items():
        ColumnarPruningMethod.apply(model.encoder[layer].rnn, 'weight_hh_l0', amount=prune_amount, dim=0)
        ColumnarPruningMethod.apply(model.encoder[layer].rnn, 'weight_ih_l0', amount=prune_amount, dim=0)
    return model

def finetune(model, epochs_per_iter):
    # keep track of losses and metrics to take the average
    train_results = dict()

    print('Finetuning')
    total_num_steps = -1
    for epoch_num in range(epochs_per_iter):
        
        loader_train = model.dataloader_train
        loader_validation = model.dataloader_validation
        # use this to restart the in case we finish all the validation data
        validation_iterator = iter(loader_validation) 
        
        start_time = time.time()
        # iterate over the train data
        for _, train_batch in enumerate(loader_train):
            
            losses, predictions = model.train_step(train_batch)
            total_num_steps += 1
            
            for k, v in losses.items():
                if k not in train_results.keys():
                    train_results[k] = list()
                train_results[k].append(v)
            
            if total_num_steps % STEPS_PER_VALIDATION == 0:
                # calculate accuracy for the training only here
                # since doing for every batch is expensive and slow...
                time_taken = time.time() - start_time
                validate(train_batch, loader_validation, validation_iterator, epoch_num, total_num_steps, predictions, train_results, time_taken)
                start_time = time.time()
                train_results = dict()

    return model

def validate(train_batch, loader_validation, validation_iterator, epoch_num, total_num_steps, predictions, train_results, time_taken):
    # calculate accuracy for the training only here since doing for every batch
    # is expensive and slow...
    predictions_decoded = model.decode(predictions, greedy = True)
    metrics = model.evaluate(train_batch, predictions_decoded)
    
    # log the train results
    log_df = generate_log_df(list(train_results.keys()), list(metrics.keys()))
    for k, v in train_results.items():
        log_df[k + '.train'] = np.mean(v)
    for k, v in metrics.items():
        log_df[k + '.train'] = np.mean(v)
    
    try:
        validation_batch = next(validation_iterator)
    except StopIteration:
        validation_iterator = iter(loader_validation)
        validation_batch = next(validation_iterator)
                    
    # calculate and log the validation results
    losses, predictions = model.validation_step(validation_batch)
    predictions_decoded = model.decode(predictions, greedy = True)
    metrics = model.evaluate(validation_batch, predictions_decoded)
    
    for k, v in losses.items():
        log_df[k + '.val'] = v # do not need the mean as we only did it once
    for k, v in metrics.items():
        log_df[k + '.val'] = np.mean(v)
        
    # calculate time it took since last validation step
    log_df['epoch'] = str(epoch_num)
    log_df['step'] = str(total_num_steps)
    log_df['time'] = int(time_taken)
    for param_group in model.optimizer.param_groups:
        log_df['lr'] = param_group['lr']
        
    # save the model if we are at a saving step
    log_df['checkpoint'] = 'no'
    if total_num_steps % STEPS_PER_CHECKPOINT == 0:
        log_df['checkpoint'] = 'yes'
        model.save(os.path.join(checkpoints_dir, 'checkpoint_' + str(int(total_num_steps / STEPS_PER_CHECKPOINT)) + '.pt'))
    
    # write to log
    if not os.path.isfile(os.path.join(output_dir, 'train.log')):
        log_df.to_csv(os.path.join(output_dir, 'train.log'), header=True, index=False)
    else: # else it exists so append without writing the header
        log_df.to_csv(os.path.join(output_dir, 'train.log'), mode='a', header=False, index=False)
        
    # write results to console
    print(log_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help='Checkpoint file to prune and finetune')
    parser.add_argument("--data-dir", type=str, required=True, help='Path where the data for the dataloaders is stored')
    parser.add_argument("--output-dir", type=str, required=True, help='Path where the model is saved')
    # parser.add_argument("--model", type=str, choices=['bonito', 'catcaller', 'causalcall', 'mincall', 'sacall', 'urnano', 'halcyon',], help='Model')
    parser.add_argument("--window-size", type=int, choices=[400, 1000, 2000, 4000], default = 2000, help='[UNUSED] Window size for the data')
    parser.add_argument("--num-iterations", type=int, default = 2)
    parser.add_argument("--prune", metavar='LAYER=PRUNE_AMT', nargs='+', default=[], help='Amount to prune a layer per iteration. Example: 4=0.5 3=0.2')
    parser.add_argument("--epochs-per-iter", type=int, default = 2, help='Number of finetuning epochs per pruning iteration')
    parser.add_argument("--batch-size", type=int, default = 64)
    parser.add_argument("--starting-lr", type=float, default = 0.0005) #was 0.001
    parser.add_argument("--warmup-steps", type=int, default = 5000)
    # parser.add_argument("--use-scaler", action='store_true', help='use 16bit float precision')
    parser.add_argument("--overwrite", action='store_true', help='delete existing files in folder')
    parser.add_argument("--selfish-rnn", action='store_true', help='use Selfish RNN pruning method')
    add_sparse_args(parser)
    args = parser.parse_args()

    print('Debug: prune config string:', args.prune)
    prune_config = { int(k):float(v) for k,v in dict(map(lambda s: s.split('='), args.prune)).items() }
    print('Prune config:')
    for k, v in prune_config.items():
        print('Prune layer {} by {}'.format(k, v))

    print('Creating dataset')
    dataset = BaseNanoporeDataset(
        data_dir = args.data_dir, 
        decoding_dict = NON_RECURRENT_DECODING_DICT, 
        encoding_dict = NON_RECURRENT_ENCODING_DICT, 
        split = 0.95, 
        shuffle = True, 
        seed = 1,
    )

    dataloader_train = DataLoader(
        dataset, 
        batch_size = args.batch_size, 
        sampler = dataset.train_sampler, 
        num_workers = 4
    )
    dataloader_validation = DataLoader(
        dataset, 
        batch_size = args.batch_size, 
        sampler = dataset.validation_sampler, 
        num_workers = 4
    )

    print('Creating model')
    ##   MODEL PART 1       ##
    model = Model(
        load_default = True,
        device = device,
        dataloader_train = dataloader_train,
        dataloader_validation = dataloader_validation,
    )
    model = model.to(device)
    
    print('Creating optimization')
    ##    OPTIMIZATION     ##
    optimizer = torch.optim.Adam(model.parameters(), lr=args.starting_lr)
    total_steps =  (len(dataset.train_idxs)*args.epochs_per_iter)/args.batch_size
    cosine_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,total_steps, eta_min=0.00001, last_epoch=-1, verbose=False)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier = 1.0, total_epoch = args.warmup_steps, after_scheduler=cosine_lr)
    schedulers = {'lr_scheduler': cosine_lr}
    clipping_value = 2
    use_sam = False

    ##   MODEL PART 2       ##
    model.optimizer = optimizer
    model.schedulers = schedulers
    model.clipping_value = clipping_value
    model.use_sam = use_sam

    model.load(args.checkpoint, initialize_lazy = True)
    for g in model.optimizer.param_groups:
        g['lr'] = args.starting_lr
    model = model.to(device)

    model.mask = None
    ## SELFISH RNN ##
    if args.selfish_rnn:
        decay = CosineDecay(args.death_rate, total_steps)
        model.mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay, growth_mode=args.growth, redistribution_mode=args.redistribution)
        model.mask.prune_every_k_steps = 31000
        model.mask.add_module(model, sparse_init=args.sparse_init, density=args.density)

    print('Creating outputs')
    # output stuff
    output_dir = args.output_dir
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')

    # check output dir
    if not os.path.isdir(output_dir) or args.overwrite:
        shutil.rmtree(output_dir, ignore_errors=True)
        os.mkdir(output_dir)
        os.mkdir(checkpoints_dir)
    else:
        if len(os.listdir(output_dir)) > 0:
            raise FileExistsError('Output dir contains files')
        else:
            os.mkdir(checkpoints_dir)

    main(model, args.num_iterations, prune_config, args.epochs_per_iter)
