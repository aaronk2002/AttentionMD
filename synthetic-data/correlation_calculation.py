import torch
import numpy as np
import argparse
from tools.bregman_div import bregman_correlation

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--att_svm", help="result file to get att-svm solution", type=str)
parser.add_argument("--Ws", help="result file to get Ws history", type=str)
parser.add_argument("-p", help="the lp", type=float)
parser.add_argument("--output", help="output destination", type=str)
args = parser.parse_args()

# Extract data
Ws = torch.load(args.Ws)["Ws"]
att_svm = torch.load(args.att_svm)["att-svm"]
epochs = len(Ws)

# Get correlation
W_corrs = np.zeros((epochs,))
for it in range(epochs):
    W_corrs[it] = bregman_correlation(att_svm, Ws[it], args.p)

# Save
torch.save(W_corrs, args.output)
