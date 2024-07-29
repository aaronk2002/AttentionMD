import torch
import argparse
from tools.generate_data import data_generator
from tools.train_small import full_train

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-n", help="An integer representing number of samples", type=int)
parser.add_argument(
    "-T", help="An integer representing number of tokens per samples", type=int
)
parser.add_argument(
    "-d", help="An integer representing the dimensionality of each tokens", type=int
)
parser.add_argument("-p", help="A float representing the type of SMD", type=float)
parser.add_argument("--lr", help="A float representing learning rate", type=float)
parser.add_argument(
    "--epochs", help="An integer representing number of epochs", type=int
)
parser.add_argument("--seed", help="The seed for randomness", type=int)
parser.add_argument(
    "--normalized",
    help="Whether the MD is normalized",
    type=bool,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--parameterization", help="The parameterization that is used", type=str
)
parser.add_argument(
    "--std", help="standard dev of initialization", type=float, default=0
)
parser.add_argument("--output", help="The result destination", type=str)
args = parser.parse_args()

# Get arguments
n = args.n
T = args.T
d = args.d
p = args.p
lr = args.lr
epochs = args.epochs
seed = args.seed
normalized = args.normalized
parameterization = args.parameterization
std = args.std
device = "cuda" if torch.cuda.is_available() else "cpu"
print(
    f"n = {n}, T = {T}, d = {d}, p = {p}, lr = {lr}, epochs = {epochs}, seed = {seed}, parameterization = {parameterization}, normalized = {normalized}, device = {device}, std = {std}"
)

# data generation
X, Y, Z, v = data_generator(n, T, d, seed)
X, Y, Z, v = X.double(), Y.double(), Z.double(), v.double()
X, Y, Z, v = X.to(device), Y.to(device), Z.to(device), v.to(device)
# training
result = full_train(
    X, Y, Z, v, epochs, lr, p, normalized, parameterization, device, std
)
torch.save(result, args.output)
