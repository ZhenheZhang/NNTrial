#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
@File   :   ak_bigram.py
@Author :   Zhenhe Zhang
@Date   :   2024/1/29
@Notes  :   Andrej Karpathy's Repo: nanogpt-lecture@https://github.com/karpathy/ng-video-lecture

"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse


# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


# NOT USED
def split_dataset(csv_file: str, ratio: float, split: str):
    import random
    random_seed = 42
    random.seed(random_seed)

    annotation_data1 = []
    annotation_data2 = []
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip the header: split, topic, label, fine-grained-label, TEXT
        for row in reader:
            if random.random() <= ratio:
                annotation_data1.append(row)
            else:
                annotation_data2.append(row)
    if split == 'train':
        csv_writer(annotation_data1, 'train.csv')
        csv_wreter(annotation_data2, 'eval.csv')
    elif split == 'eval':
        csv_writer(annotation_data1, 'test.csv')
        csv_wreter(annotation_data2, 'dev.csv')
    else:
        raise RuntimeError("split NOT SUPPORTED: {}".format(split))
    return


def run(args):
    # PART1: hyperparameters
    batch_size = args.batch_size
    block_size = args.block_size
    max_iters = args.max_iters
    eval_iters = args.eval_iters
    eval_interval = args.eval_interval
    learning_rate = args.learning_rate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1337)


    # PART2: data loading
    input_path = args.input
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]


    def get_batch(split):
        # generate a small batch of data of inputs x and targets y
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out


    # PART3: create model
    model = BigramLanguageModel(vocab_size)
    m = model.to(device)

    # PART4:  create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


    # PART5: create training loop
    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # sample a batch of data
        xb, yb = get_batch('train')
        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(f"\n<Generated>: {decode(m.generate(context, max_new_tokens=500)[0].tolist())}")



def main(argv=None):
    # Init agrparse
    parser = argparse.ArgumentParser()
    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    parser.add_argument('--input', type=str, default='../data/tinyshakespeare_input.txt', help='input text')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--block_size', type=int, default=8, help='block size')
    parser.add_argument('--max_iters', type=int, default=3000, help='Maximum training number of Iterations')
    parser.add_argument('--eval_iters', type=int, default=200, help='Maximum evaluation number of Iterations')
    parser.add_argument('--eval_interval', type=int, default=300, help='Evaluation interval')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--num_gpus', type=int, default=0, help='number of gpus')
    parser.add_argument('--num_nodes', type=int, default=1, help='number of nodes')
    args, script_args = parser.parse_known_args(args=argv)
    # NOT USED: script_args
    run(args)



if __name__ == '__main__':
    main()
