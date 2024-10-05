# Real Data Experiments for "Optimizing Attention with Mirror Descent: Generalized Max-Margin Token Selection" Paper

This folder contains the code for the experiments that uses the [Stanford Large Movie Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).
The experiment requires you to first download the dataset from their website as a .tar file.
Once you have downloaded the .tar file, unzip and move the `test` and `train` folder into a new folder called `dataset/IMDb`.
This experiment will train several models and save them in the `results` folder.

To run the experiment, simply run the bash scripts in the `bash_scripts` folder, each of them trains 10 models with randomized initialization using either $\ell_{1.1}$, $\ell_2$, or $\ell_3$-MD and one of the three different model sizes.
There are 9 possible combination of algorithm and model sizes, hence 9 different bash scripts. Each of the resulting model and training and testing losses and accuracy history are stored in `results`.
You can then get the average and standard deviation test accuracy for each of the 9 different settings using `test_acc.ipynb`, visualize the histogram of parameters
for the model using `histogram.ipynb`, and the resulting attention map using `attn_map.ipynb`.

We recommend using GPUs to run the training scripts as each epoch could take around half an hour to run when we only used CPUs. We used V100 GPUs specifically, and we acquired them from [MIT Supercloud](https://supercloud.mit.edu/)

## Additional Details

These scripts run the `train.py` python script using configurations from `config`, and if you would like to use different configurations, here are some of the configuration parameters that you can tinker around with along with their explanations:
- `n_embd`: int, the dimensionality of token embeddings
- `n_blocks`: int, the number of attention blocks in the model
- `n_head`: int, the number of attention heads in each blocks
- `n_hidden`: int, the number of hidden state in the MLP portion of each attention block
- `vocab_size`: int, the number of tokens in the vocab
- `max_length`: int, the maximum length that the model can take
- `epochs`: int, the maximum number of training cycles that the model will go through, the actual number of training loop can be smaller if the training accuracy reaches `train_acc_limit` first
- `dropout`: float between 0 and 1, the dropout rate
- `lr`: float, the step size
- `bias`: bool, whether the linear layers use bias
- `from_prev_results`: bool, whether we train from an existing result, if `True` then `prev_result_filename` must point to a result file
- `dataset`: str, the dataset folder inside of the `dataset` directory, currently only supports `"IMDb"`
- `prev_result_filename`: str, ignored if `from_prev_results` is `False`, otherwise it is where we get the pre-trained model that we wish to fine-tune
- `outfile`: str, the address to store the results, must be a folder if `repeat` is not None
- `batch_size`: int, the batch size
- `p`: float > 1, the $p$ used for the $\ell_p$ algorithms and SVMs
- `repeat`: int or `None`, if `None` then we train one instance of the model, if it is an integer then we train that many number of models independently and store them in the `outfile` folder
- `train_acc_limit`: float between 0 and 1, stops training when training accuracy reaches this value

The tokenizer is in the `tokenizer` folder, which was downloaded using the `tokenizer_downloader.ipynb` notebook.

Certain parts of the transformer model used here are adapted from the [nanoGPT](https://github.com/karpathy/nanoGPT) repository.
