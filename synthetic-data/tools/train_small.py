import torch
import numpy as np
from tqdm import tqdm
from tools.optimizers import SMD
from tools.small_model import AttWModel, AttKQModel
from tools.att_svm import att_svm_solver, att_svm_solver_nuc, joint_wv_att_svm_solver
from tools.bregman_div import bregman_correlation

eps = torch.finfo(torch.double).eps


def full_train(
    X, Y, z, v, epochs, lr, p, normalized, parameterization, device, std=0.01
):
    """
    Given the sequence of tokesn X, the cross tokens z, the labels Y,
    train a model with lr as learning rate using l_p MD with epochs training
    loops. normalized is boolean determining if we use normalized MD.
    Returns a dictionary with att-svm solution, loss history, parameter
    history, and correlation history
    """
    n, T, d = X.size()

    # Create model
    model = (
        AttWModel(d, std).double()
        if parameterization == "W" or parameterization == "VW"
        else AttKQModel(d).double()
    )
    model = model.to(device)
    model.v.data = v
    params = (
        [{"params": [model.W.weight], "lr": lr}]
        if parameterization == "W"
        else (
            [{"params": [model.K.weight, model.Q.weight], "lr": lr}]
            if parameterization == "KQ"
            else [{"params": [model.W.weight, model.v], "lr": lr}]
        )
    )

    # Train
    optimizer = SMD(params, p, normalized)
    Ws = np.zeros((epochs + 1, d, d))
    vs = np.zeros((epochs + 1, d))
    losses = np.zeros((epochs,))
    softmax_prob = np.zeros((epochs,))
    logistic_prob = np.zeros((epochs,))
    Ws[0] = (
        (
            model.W.weight.detach().T
            if parameterization == "W" or parameterization == "VW"
            else model.Q.weight.T.mm(model.K.weight).T.detach()
        )
        .to("cpu")
        .numpy()
    )
    vs[0] = model.v.data.detach().to("cpu").numpy()
    for it in tqdm(range(epochs)):

        # Zero out gradient
        for param in model.parameters():
            param.grad = None

        # Loss calculation
        out = model(X, z).view(-1)
        loss = torch.log(1 + torch.exp(-Y * out))
        loss = loss.mean()

        # Step Optimizer
        loss.backward()
        optimizer.step()

        # Record data
        W = (
            model.W.weight.detach().T
            if parameterization == "W" or parameterization == "VW"
            else model.Q.weight.T.mm(model.K.weight).T.detach()
        )
        W = W.to("cpu").numpy()
        sfx_out = model.sfx_out.detach().max(dim=-1)
        ids = sfx_out[1]
        softmax_prob[it] = sfx_out[0].numpy().mean()
        Ws[it + 1] = W
        vs[it + 1] = model.v.data.detach().to("cpu").numpy()
        losses[it] = loss.item()

        # Logistic probability calculation
        logistic_prob[it] = 0
        for sample in range(n):
            logistic_prob[it] += 1 / (
                1
                + np.exp(
                    -Y[sample] * X[sample, ids[sample]].reshape(1, -1) @ vs[it + 1]
                )
            )
        logistic_prob[it] /= n

    # Att-SVM
    X, z, ids = X.to("cpu").numpy(), z.to("cpu").numpy(), ids.to("cpu").numpy()
    Y = Y.to("cpu").numpy()
    sol_att_svm = (
        att_svm_solver(X, z, ids, p)
        if parameterization == "W"
        else (
            att_svm_solver_nuc(X, z, ids)
            if parameterization == "KQ"
            else joint_wv_att_svm_solver(X, Y, z, ids, p)
        )
    )

    return {
        "att-svm": sol_att_svm,
        "Ws": Ws,
        "vs": vs,
        "losses": losses,
        "sfx_prob": softmax_prob,
        "log_prob": logistic_prob,
        "alpha": ids,
    }
