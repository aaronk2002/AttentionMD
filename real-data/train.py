import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import copy
from time import time
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

from tools.model import TransformerClassifier
from tools.dataset_classes import IMDb
from tools.train_config import TrainConfig
from tools.optimizer import SMD


def get_dataset(dataset, max_length=512, device="cpu"):
    """
    Get the dataset
    """
    if dataset == "IMDb":
        return IMDb(True, max_length, device), IMDb(False, max_length, device)
    else:
        raise NotImplementedError()


def train_model(config):
    """
    Train a model and save the result
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"config = {str(config)}")

    # Get the dataset
    start_time = time()
    train_set, test_set = get_dataset(config.dataset, config.max_length, device)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=True)
    n_train, n_test = len(train_set), len(test_set)
    elapsed = time() - start_time
    sec = round(elapsed % 60)
    elapsed = elapsed // 60
    minutes = elapsed % 60
    hrs = elapsed // 60
    print(f"Dataloading takes {hrs} hours {minutes} minutes {sec} seconds")

    # Get the model
    initial_config = {}
    if config.from_prev_result:
        initial_config = torch.load(config.prev_result_filename)
    else:
        initial_config = {
            att: torch.empty((0,))
            for att in ["train_loss", "train_acc", "test_loss", "test_acc"]
        }
        initial_config["model"] = TransformerClassifier(config)

    n_trials = config.repeat
    if n_trials is None:
        n_trials = 1
    for trial in range(n_trials):
        print(f"Trial {trial+1}")

        # Training set-up
        model = copy.deepcopy(initial_config["model"].to(device))
        train_loss = initial_config["train_loss"].tolist()[:]
        train_acc = initial_config["train_acc"].tolist()[:]
        test_loss = initial_config["test_loss"].tolist()[:]
        test_acc = initial_config["test_acc"].tolist()[:]
        optimizer = SMD(
            [{"params": list(model.parameters()), "lr": config.lr}], p=config.p
        )
        loss_fn = torch.nn.BCELoss(reduction="sum")

        # Training Loop
        for idx in range(config.epochs):

            # New stats
            train_loss.append(0.0)
            train_acc.append(0.0)
            test_loss.append(0.0)
            test_acc.append(0.0)

            # Step
            print(f"Epoch {idx+1}")
            model.train()
            for X, y in tqdm(train_loader, desc="  Training"):
                optimizer.zero_grad()
                prediction = model(X, device)
                loss = loss_fn(prediction, y.float())
                acc = torch.sum((prediction > 0.5) == y)
                train_loss[-1] += loss.item()
                train_acc[-1] += acc.item()
                loss.backward()
                optimizer.step()
            train_loss[-1] /= n_train
            train_acc[-1] /= n_train
            print(
                f"    Training Loss = {train_loss[-1]} Training Accuracy = {train_acc[-1]}"
            )

            # Evaluate
            model.eval()
            for X, y in tqdm(test_loader, desc="  Testing"):
                prediction = model(X, device)
                loss = loss_fn(prediction, y.float())
                acc = torch.sum((prediction > 0.5) == y)
                test_loss[-1] += loss.item()
                test_acc[-1] += acc.item()
            test_loss[-1] /= n_test
            test_acc[-1] /= n_test
            print(f"    Test Loss = {test_loss[-1]} Test Accuracy = {test_acc[-1]}")

            # Stopping condition
            if train_acc[-1] >= (config.train_acc_lim or 1.0):
                break

        # Save result
        result = {
            "model": model,
            "train_loss": torch.tensor(train_loss),
            "train_acc": torch.tensor(train_acc),
            "test_loss": torch.tensor(test_loss),
            "test_acc": torch.tensor(test_acc),
        }
        if config.repeat is None:
            torch.save(result, config.outfile)
        else:
            torch.save(result, f"{config.outfile}/{trial}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config file address", type=str)
    args = parser.parse_args()
    print(f"Training with config {args.config}")
    train_model(TrainConfig(args.config))
