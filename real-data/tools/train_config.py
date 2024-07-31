import json


class TrainConfig:

    def __init__(self, filename):
        data = json.load(open(filename))
        self.n_embd = data["n_embd"]
        self.n_blocks = data["n_blocks"]
        self.bias = data["bias"]
        self.n_hidden = data["n_hidden"]
        self.dropout = data["dropout"]
        self.n_head = data["n_head"]
        self.vocab_size = data["vocab_size"]
        self.max_length = data["max_length"]
        self.epochs = data["epochs"]
        self.lr = data["lr"]
        self.from_prev_result = data["from_prev_result"]
        self.dataset = data["dataset"]
        self.prev_result_filename = data["prev_result_filename"]
        self.outfile = data["outfile"]
        self.misc = data["misc"]
        self.batch_size = data["batch_size"]
        self.p = data["p"]
        self.repeat = data["repeat"]
        self.final_rep_agg = "mean"
        if "final_rep_agg" in data:
            self.final_rep_agg = data["final_rep_agg"]
        self.scale = None
        if "scale" in data:
            self.scale = data["scale"]
        self.mask = False
        if "mask" in data:
            self.mask = data["mask"]
        self.train_acc_lim = None
        if "train_acc_lim" in data:
            self.train_acc_lim = data["train_acc_lim"]

    def __str__(self):
        return (
            f"\nn_embd: {self.n_embd}"
            + f"\nn_blocks: {self.n_blocks}"
            + f"\nn_blocks: {self.n_blocks}"
            + f"\nbias: {self.bias}"
            + f"\nn_hidden: {self.n_hidden}"
            + f"\ndropout: {self.dropout}"
            + f"\nn_head: {self.n_head}"
            + f"\nvocab_size: {self.vocab_size}"
            + f"\nmax_length: {self.max_length}"
            + f"\nepochs: {self.epochs}"
            + f"\nlr: {self.lr}"
            + f"\nfrom_prev_result: {self.from_prev_result}"
            + f"\ndataset: {self.dataset}"
            + f"\nprev_result_filename: {self.prev_result_filename}"
            + f"\noutfile: {self.outfile}"
            + f"\nmisc: {self.misc}"
            + f"\nbatch_size: {self.batch_size}"
            + f"\np: {self.p}"
            + f"\nrepeat: {self.repeat}"
            + f"\nfinal_rep_agg = {self.final_rep_agg}"
            + f"\nscale = {self.scale}"
            + f"\nmask = {self.mask}"
            + f"\ntrain_acc_lim = {self.train_acc_lim}"
        )
