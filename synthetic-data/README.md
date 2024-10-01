# Synthetic Data Experiments for "Optimizing Attention with Mirror Descent: Generalized Max-Margin Token Selection" Paper

This folder contains the synthetic data experiments, which consists of the experiments that shows the convergence of the $\ell_p$-AttGD algorithm and that of the $\ell_p$-JointGD algorithm. For the $\ell_p$-AttGD experiments, one can run them by simply running the `run-local-1_75.sh`, `run-local-2.sh`, and `run-local-3.sh` in `bash_scripts`, which trains attention models on different synthetic datasets, which produces a history of the model parameters for each of these datasets when it was trained using $\ell_{1.75}$, $\ell_2$ (same as GD in this case), and $\ell_3$ potential functions. These scripts also provide other metrics such as the locally optimal direction that solves the corresponding $\ell_p$-AttSVM problem. All these results are stored in `result/convergence`.

To get the directional Bregman divergence, as shown in Figure 4 of the paper, run `run-correlations-W.sh`, which gets us the correlations and saves it in `result/correlation`.

Finally, `run-joint-1_75.sh`, `run-joint-2.sh`, and `run-joint-3.sh` trains the model using $\ell_p$-JointGD on the synthetic data for this training setting as described in the paper and saves the models in `result/joint_convergence`.

The above bash scripts presents the results in the result directory, which can be visualized in the `plotter.ipynb` file.

The convex optimization code used to solve the AttSVM problem are adapted from the [Transformers as Support Vector Machine](https://github.com/umich-sota/TF-as-SVM) code repo.
