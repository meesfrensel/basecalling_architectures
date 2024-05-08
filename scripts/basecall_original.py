import os
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
import argparse

from classes import BasecallerImpl, BaseFast5Dataset

import pandas as pd
import numpy as np
import torch
import torch.nn.utils.prune as prune

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print("Starting")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=[
        'bonito',
        'catcaller',
        'causalcall',
        'mincall',
        'sacall',
        'urnano',
        'halcyon',
        'own',
    ], required = True)
    parser.add_argument("--fast5-dir", type=str, help='Path to fast5 files', default = None)
    parser.add_argument("--fast5-list", type=str, help='Path to file with list of files to be processed', default = None)
    parser.add_argument("--checkpoint", type=str, help='checkpoint file to load model weights', required = True)
    parser.add_argument("--output-file", type=str, help='output fastq file', required = True)
    parser.add_argument("--chunk-size", type=int, default = 2000)
    parser.add_argument("--window-overlap", type=int, default = 200)
    parser.add_argument("--batch-size", type=int, default = 64)
    parser.add_argument("--beam-size", type=int, default = 1)
    parser.add_argument("--beam-threshold", type=float, default = 0.1)
    parser.add_argument("--model-stride", type=int, default = None)
    
    args = parser.parse_args()


    file_list = list()
    if args.fast5_dir is not None:
        if args.fast5_dir.endswith('.fast5'):
            file_list.append(args.fast5_dir)
        else:
            for f in os.listdir(args.fast5_dir):
                if f.endswith('.fast5'):
                    file_list.append(os.path.join(args.fast5_dir, f))
    elif args.fast5_list is not None:
        with open(args.fast5_list, 'r') as f:
            for line in f:
                file_list.append(line.strip('\n'))
    else:
        raise ValueError('Either --fast5-dir or --fast5-list must be given')

    print('Found ' + str(len(file_list)) + ' files')

    fast5_dataset = BaseFast5Dataset(fast5_list= file_list, buffer_size = 1)

    output_file = args.output_file

    # load model
    checkpoint_file = args.checkpoint
    
    use_amp = False
    scaler = None

    if args.model == 'halcyon':
        from halcyon.model import HalcyonModelS2S as Model # pyright: ignore[reportMissingImports]
        args.model_stride = 1
    elif args.model == 'bonito':
        from bonito.model import BonitoModel as Model # pyright: ignore[reportMissingImports]
    elif args.model == 'catcaller':
        from catcaller.model import CATCallerModel as Model # pyright: ignore[reportMissingImports]
    elif args.model == 'causalcall':
        from causalcall.model import CausalCallModel as Model # pyright: ignore[reportMissingImports]
    elif args.model == 'mincall':
        from mincall.model import MinCallModel as Model # pyright: ignore[reportMissingImports]
    elif args.model == 'sacall':
        from sacall.model import SACallModel as Model # pyright: ignore[reportMissingImports]
    elif args.model == 'urnano':
        from urnano.model import URNanoModel as Model # pyright: ignore[reportMissingImports]
    elif args.model == 'own':
        from own.model import OwnModel as Model # pyright: ignore[reportMissingImports]

    model = Model(
        load_default = True,
        device = device,
        dataloader_train = None, 
        dataloader_validation = None, 
        scaler = scaler,
        use_amp = use_amp,
    )
    model = model.to(device)

    model = model.to(device)
    for i in range(5):
        prune.identity(model.encoder[i].rnn, 'weight_ih_l0')
        prune.identity(model.encoder[i].rnn, 'weight_hh_l0')
    model.load(checkpoint_file, initialize_lazy = True)
    # for i in range(5):
    #     prune.remove(model.encoder[i].rnn, 'weight_ih_l0')
    #     prune.remove(model.encoder[i].rnn, 'weight_hh_l0')
    model = model.to(device)
    ex_input = next(iter(fast5_dataset))['x'][0:64, :].unsqueeze(1)
    print(ex_input.shape)
    torch.onnx.export(model, ex_input.to('cuda:0'), "90sparse.onnx", export_params=True, opset_version=10, do_constant_folding=True, input_names=['input'], output_names=['output'], dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})
    sys.exit()
    # prune.remove(model.encoder[4].rnn, 'weight_ih_l0')
    # prune.remove(model.encoder[4].rnn, 'weight_hh_l0')
    model = model.to(device)
    # print(model)
    
    # model.encoder[4].rnn.weight_ih_l0 = torch.nn.Parameter(model.encoder[4].rnn.weight_ih_l0.to_sparse())
    # model.encoder[4].rnn.weight_hh_l0 = torch.nn.Parameter(model.encoder[4].rnn.weight_hh_l0.to_sparse())
    # print(model.encoder[4].rnn.weight_hh_l0)
    # model.encoder[4].rnn.flatten_parameters()
    # print(model.encoder[4].rnn.weight_ih_l0)
    #model.encoder[4].rnn.flatten_parameters()

    print('to device')
    model = model.to(device)

    basecaller = BasecallerImpl(
        dataset = fast5_dataset, 
        model = model, 
        batch_size = args.batch_size, 
        output_file = output_file, 
        n_cores = 4, 
        chunksize = args.chunk_size, 
        overlap = args.window_overlap, 
        stride = args.model_stride,
        beam_size = args.beam_size,
        beam_threshold = args.beam_threshold,
    )
    #print('starting')
    #print('device: ' + str(device))
    #torch.cuda.cudart().cudaProfilerStart()
    #torch.cuda.nvtx.range_push("region name")

    start_time = time.time()

    basecaller.basecall(verbose = True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Time elapsed: {:.1f} s ({:.1f} files/second)'.format(elapsed_time, len(file_list) / elapsed_time))
    #torch.cuda.nvtx.range_pop()
    #print('done')
