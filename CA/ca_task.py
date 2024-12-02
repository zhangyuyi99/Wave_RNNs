from torch import nn, optim
import torch
import rnn_model as model_rnn
import conv_rnn_model as model_conv_rnn
import torch.nn.utils
import utils
import utils_ca
import argparse
from tqdm import tqdm
from pathlib import Path
import os
from sys import exit
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--run_name', type=str, default=None,
                    help='name of run for wandb')
parser.add_argument('--dataset', type=str, default='copy',
                    help='dataset name for wandb')
parser.add_argument('--model_type', type=str, default='wrnn',
                    help='type of model, rnn, wrnn')
parser.add_argument('--n_hid', type=int, default=100,
                    help='hidden size of recurrent net')
parser.add_argument('--rule_number', type=int, default=30,
                    help='rule number')
parser.add_argument('--num_train_samples', type=int, default=256,
                    help='number of training samples')
parser.add_argument('--num_test_samples', type=int, default=100,
                    help='number of testing samples')
parser.add_argument('--seq_size', type=int, default=50,
                    help='length of sequences')
parser.add_argument('--max_steps', type=int, default=10000,
                    help='max learning steps')
parser.add_argument('--log_interval', type=int, default=1000,
                    help='log interval')
parser.add_argument('--batch', type=int, default=128,
                    help='batch size')
parser.add_argument('--batch_test', type=int, default=50,
                    help='size of test set')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--n_ch', type=int, default=1,
                    help='Num hidden state channels')
parser.add_argument('--ksize', type=int, default=3,
                    help='Hidden Kernelsize')
parser.add_argument('--act', type=str, default='relu',
                    help='hidden state activation')
parser.add_argument('--init', type=str, default='eye',
                    help='initalization for RNN')
parser.add_argument('--freeze_rnn', type=str, default='no',
                    help='Make Recurrent weights untrained')
# parser.add_argument('--one_hot', type=str, default='True',
#                     help='Make Recurrent weights untrained')
parser.add_argument('--only_last', type=str, default='False',
                    help='Only train on last ouputs')
parser.add_argument('--freeze_encoder', type=str, default='no',
                    help='Make Encoder weights untrained')
parser.add_argument('--grad_clip', type=float, default=0.0)
parser.add_argument('--is_sweep', type=str, default='no')
parser.add_argument('--solo_init', type=str, default='no')
parser.add_argument('--patience_init', type=int, default=-1)
parser.add_argument('--patience', type=int, default=5)


args = parser.parse_args()


n_inp = 1
n_out = 1

is_sweep = utils.str_to_bool(args.is_sweep)

model_select = {'rnn': model_rnn, 'wrnn': model_conv_rnn}

model = model_select[args.model_type].coRNN(n_inp, args.n_hid, n_out, args.n_ch, args.act, args.ksize,
                                            args.init, args.freeze_rnn, args.freeze_encoder, args.solo_init).to(device)

import wandb
wandb.init(name=args.run_name,
            project='CA', 
            # entity='ENTITY_NAME', 
            # dir='WANDB_DIR',
            config=args)
            
wandb.watch(model)

def log(key, val):
    print(f"{key}: {val}")
    wandb.log({key: val})

# objective = nn.CrossEntropyLoss()
objective = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

fname = f'result/adding_test_log_{args.model_type}_h{args.n_hid}_seq_size{args.seq_size}.txt'

def test():
    model.eval()
    with torch.no_grad():
        # data, label = utils.get_batch(args.batch_test, args.seq_size)
        data, label = utils_ca.get_batch_ca(args.rule_number, args.num_test_samples, args.seq_size)
        label = label
        out, _ = model(data.to(device))
        if utils.str_to_bool(args.only_last) == True:
            out = out[-args.mem_len:]
            label = label[-args.mem_len:]
        loss = objective(out.reshape(-1, n_inp), label.to(device).reshape(-1).long())
        loss = loss / float(args.batch_test)

    return loss.item()


test_mse = []
best_mse = 1e10
flat_steps = 0
for i in tqdm(range(args.max_steps), desc=f"Copy_{args.model_type}_h{args.n_hid}, T{args.seq_size}", disable=is_sweep):
    # data, label = utils.get_batch(args.batch, args.seq_size)
    data, label = utils_ca.get_batch_ca(args.rule_number, args.num_train_samples, args.seq_size)
    label = label
    
    optimizer.zero_grad()

    out, seq  = model(data.to(device), get_seq=True)

    if utils.str_to_bool(args.only_last) == True:
        out = out[-args.mem_len:]
        label = label[-args.mem_len:]
    # print("out shape:", out.shape)
    # print("label shape:", label.shape)

    loss = objective(out.reshape(-1, n_inp).squeeze(-1), label.to(device).reshape(-1).float())
    loss = loss / float(args.batch)
    loss.backward()
    
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

    optimizer.step()

    if(i%args.log_interval==0 and i!=0):
        log('Train Loss:', loss)

        mse_error = test()
        log('Test MSE:', mse_error)
        test_mse.append(mse_error)
        
        if mse_error <= best_mse:
            best_mse = mse_error

        # utils.plot_output(data[:, 0], out[:, 0], label[:, 0])
        utils.plot_output(data, out.squeeze(), label)

        utils.Plot_Vid(model(data.to(device), get_seq=True)[1][:, 0], fps=args.seq_size // 5)
        if torch.isnan(loss):
            exit()

        model.train()

        Path('result').mkdir(parents=True, exist_ok=True)
        f = open(fname, 'a')
        f.write('test mse: ' + str(round(test_mse[-1], 2)) + '\n')
        f.close()

        if torch.isnan(loss):
            if not solved:
                log('Solved Iter', args.max_steps)
            exit()

if not solved:
    log('Solved Iter', args.max_steps)
