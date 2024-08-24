# Real Data Experiments for "Optimizing Attention with Mirror Descent: Generalized Max-Margin Token Selection" Paper

This folder contains the code for the experiments that uses the [Stanford Large Movie Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).
To run the experiments, you must first download the dataset from their website and create a new folder `dataset/imdb_larger`.
Once you have downloaded the .tar file, unzip and move the `test` and `train` folder into `dataset/imdb_larger`, now you are ready to run the experiment.

To run the experiment, simply run the bash scripts in the `bash_scripts` folder, each of them trains 10 models with randomized initialization using either $\ell_{1.1}$, $\ell_2$, or $\ell_3$-MD and one of the three token representation aggregation methods described in the paper.
There are 9 possible combination of algorithm and aggregation, hence 9 different bash scripts. Each of the resulting model and training and testing losses and accuracy history are stored in `results`.
You can then get the average and standard deviation test accuracy for each of the 9 different settings using `test_acc.ipynb`, visualize the histogram of parameters
for the model using `histogram.ipynb`, and the resulting attention map using `attn_map.ipynb`.

Certain parts of the transformer model used here are adapted from the [nanoGPT](https://github.com/karpathy/nanoGPT) repository.
