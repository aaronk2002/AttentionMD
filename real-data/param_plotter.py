import torch
import argparse
import matplotlib.pyplot as plt

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model file address", type=str, required=False)
parser.add_argument("--param", help="param file address", type=str, required=False)
parser.add_argument(
    "--newparam", help="save address for param", type=str, required=False
)
parser.add_argument("--img", help="save address for image", type=str)
parser.add_argument("--bins", help="number of bins", type=int)
args = parser.parse_args()

# Get parameters
parameters = torch.tensor([])
if args.param is None:
    result = torch.load(args.model, map_location=torch.device("cpu"))
    for param in result["model"].parameters():
        parameters = torch.cat((parameters, param.flatten()))
    torch.save(parameters, args.newparam)
else:
    parameters = torch.load(args.param)

# Get image
hist = torch.histogram(parameters, args.bins)
plt.plot(hist.bin_edges[:-1].detach().numpy(), hist.hist.detach().numpy())
plt.savefig(args.img)
