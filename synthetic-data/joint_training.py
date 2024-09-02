import torch
import argparse
from tools.train_small import full_train

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-p", help="A float representing the type of SMD", type=float)
parser.add_argument("--lr", help="A float representing learning rate", type=float)
parser.add_argument(
    "--epochs", help="An integer representing number of epochs", type=int
)
parser.add_argument("--output", help="The result destination", type=str)
args = parser.parse_args()
lr, p, epochs = args.lr, args.p, args.epochs

# Data
n, T, d = 2, 3, 2
device = "cuda" if torch.cuda.is_available() else "cpu"
X = torch.Tensor(
    [[[-2.7, 1.2], [1.4, 2.1], [1.3, -0.1]], [[0.4, -2.2], [-1.1, -0.4], [0.9, 0.1]]]
).double().to(device)
Y = torch.Tensor([1, -1]).double().to(device)
z = torch.Tensor([X[0, 0].tolist(), X[1, 0].tolist()]).double().to(device)
v = torch.zeros((d,)).double().to(device)

result = full_train(X, Y, z, v, epochs, lr, p, True, "VW", device, 0)
torch.save(result, args.output)
