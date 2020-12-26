
from reformer_pytorch import ReformerLM
from reformer_pytorch.generative_tools import TrainingWrapper

import os
import random
import tqdm
import arithmeticcoding_fast
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import time
import argparse
import contextlib
import arithmeticcoding_fast
import json
import struct
import tempfile
import shutil
import sys
# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 1
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
GENERATE_LENGTH = 512
SEQ_LEN = 4096
heads = 8
local_heads = 4
depth = 5
time_list = []
torch.manual_seed(0)
#np.random.seed(0)
# helpers
parser = argparse.ArgumentParser(description='Input')
parser.add_argument('-model', action='store', dest='model_weights_file',
					help='model file')
parser.add_argument('-model_name', action='store', dest='model_name',
					help='model file')
parser.add_argument('-batch_size', action='store', dest='batch_size', type=int,
					help='model file')
parser.add_argument('-data', action='store', dest='sequence_npy_file',
					help='data file')
parser.add_argument('-data_params', action='store', dest='params_file',
					help='params file')
parser.add_argument('-output', action='store',dest='output_file_prefix',
					help='compressed file name')
parser.add_argument('-gpu', action='store', dest='gpu_id', default="",
					help='params file')

args = parser.parse_args()

#os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n), writeable=False)

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# instantiate model
def init_model(model_path):
    torch.manual_seed(0)
    model = ReformerLM(
        dim = 512,
        depth = depth,
        max_seq_len = SEQ_LEN,
        num_tokens = 256,
        heads = heads,
        bucket_size = 64,
        n_hashes = 4,
        ff_chunks = 10,
        lsh_dropout = 0.1,
        weight_tie = True,
        causal = True,
        n_local_attn_heads = local_heads,
        use_full_attn = False # set this to true for comparison with full attention
    )
    for k in model.state_dict():
        print(k)
    print(model_path)
    print("---------------"*5)
    weight = torch.load(model_path)
    for k in weight.keys():
        print(k)
    print("---------------"*5)
    #for k in model.state_dict().keys():
    #    print(k)
    model = TrainingWrapper(model)
    model.load_state_dict(torch.load(model_path))
    model.cuda()

    return model

# prepare enwik8 data

inp = './data/enwik8.gz'

with gzip.open(inp) as file:
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    trX, vaX = np.split(X, [int(90e6)])
    data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

def set_bn_eval(model):
    pass

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

def decode(len_series, timesteps, vocab_size, compressed_file, model_path):
    series = np.zeros(len_series, dtype = np.uint8).astype('int')
    f = open(compressed_file, 'rb')
    bitin = arithmeticcoding_fast.BitInputStream(f)
    dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
    prob = np.ones(vocab_size)/vocab_size
    cumul = np.zeros(vocab_size+1, dtype = np.uint64)
    cumul[1:] = np.cumsum(prob*10000000 + 1)        
    
    decode_model = init_model(model_path)
    decode_model.eval()
    for j in range(min(timesteps,len_series)):
        series[j] = dec.read(cumul, vocab_size)
    for i in range(len_series-timesteps):
        tmp = torch.LongTensor(series[i:i+timesteps].reshape(1,-1)).cuda()
        with torch.no_grad():
            decode_model.eval()
            prob = decode_model(tmp)[:, -1, :]
            prob = F.softmax(prob).cpu().detach().numpy()
        cumul[1:] = np.cumsum(prob*10000000 + 1)
        series[i+timesteps] = dec.read(cumul, vocab_size)
    bitin.close()
    f.close()
    print(decode_tokens(series))
    return series

def encode(len_series, timesteps, startencode_stamp, vocab_size, X, y, compressed_file, model_path):
    global time_list
    f = open(compressed_file, 'wb')
    bitout = arithmeticcoding_fast.BitOutputStream(f)
    enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
    prob = np.ones(vocab_size)/vocab_size
    cumul = np.zeros(vocab_size+1, dtype = np.uint64)
    cumul[1:] = np.cumsum(prob*10000000 + 1)        
    
    model = init_model(model_path)
    model.eval()

    out_check = open(compressed_file + '_check', 'w')
    seq = []
    ts_start = startencode_stamp - timesteps
    start_time = time.process_time()
    for j in range(timesteps):
        enc.write(cumul, X[ts_start,j])
        seq.append(X[ts_start, j])

    for i in range(ts_start, ts_start+len_series-timesteps):
        tmp = torch.LongTensor(X[i:i+1, :]).cuda()
        with torch.no_grad():
            prob = model(tmp)[:, -1, :]
            prob = F.softmax(prob).detach().cpu().numpy()
        cumul[1:] = np.cumsum(prob*10000000 + 1)
        enc.write(cumul, y[i])
        seq.append(y[i][0])
    enc.finish()
    end_time = time.process_time()
    print(end_time - start_time)
    time_list.append(end_time - start_time)
    bitout.close()
    f.close()
    out_check.write(decode_tokens(seq))
    out_check.close()
    
    return

def main():

    inp = './data/enwik8.gz'

    with gzip.open(inp) as file:
        sequence = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
        trX, vaX = np.split(sequence, [int(90e6)])
        data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)

    model_path ='enwiki8_layer{0}_seq{1}_head{2}.pth'.format(depth, SEQ_LEN, heads)
    compressed_file = "enwiki8.compress.ts64_layer{0}_seq{1}_head{2}".format(depth, SEQ_LEN, heads)
    vocab_size = 256
    batch_size = 64
    timesteps = 64
    real_seq_length = 4096
    len_series = timesteps + real_seq_length
    start_timestamps = [ i*4096 for i in range(1, 101, 10) ]
    sequence = vaX.reshape(-1)
    series = sequence.copy()
    data = strided_app(series, timesteps+1, 1)
    X = data[:, :-1]
    y = data[:, -1:]
    total_size = 0
    for ts in start_timestamps:
        encode(len_series, timesteps, ts,  vocab_size, X, y, compressed_file, model_path)
        compressed_size = os.path.getsize(compressed_file)
        total_size += compressed_size
        print(compressed_size)

    print("Avg compressed size: ", total_size/len(start_timestamps))
    print("Avg compressed time: ", sum(time_list)/len(start_timestamps))
    #decode(len_series, timesteps, vocab_size, compressed_file)

if __name__ == "__main__":
     main()

