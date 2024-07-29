# Mirror Descent for Generalized Max-Margin Token Selection in Attention Mechanism

This is the codebase for the paper "Mirror Descent for Generalized Max-Margin Token Selection in Attention Mechanism". To reproduce the results, we have the `bash_scripts` directory that has several bash code to run the experiments:

- `run-local-{p}.sh` for $p=$ 2, 3, and 1_75
  These three bash scripts each trains the single-layer attention model with the $l_p$ Mirror Descent algorithm for 100 times for 100 different synthetically generated datasets, and calculates the 100 solutions for the $l_p$ -ATT-SVM problems. Once completed, it trains the model one more time for the first dataset but with the initialization being at zero. All results will be saved in the `result/convergence/p{p}` folder. The first 100 different results will be saved under the name `W{i}.pt` with $i$ being 1 to 100, while the last result with zero initialization will be named `SingleW.pt`
- `run-correlations-W.sh`
  This bash script is used to calculate the correlation coefficient between the iterates of the $l_p$ Mirror Descent algorithm saved in `result/convergence/p{p}/W{i}.pt` and the $l_q$-ATT-SVM solutions saved in `result/convergence/p{q}/W{i}.pt`, for all $p,q\in\set\{1.75,2,3\}$ and $i=1,...,100$, results are saved in `result/correlation/{q}-{p}/W{i}.pt`
- `run-joint-{p}.sh` for $p=$ 2, 3, and 1_75
  These three bash scripts each trains the single-layer attention model with the $l_p$ Mirror Descent algorithm with the parameters initialized at zero. The optimizer jointly learns both the $W$ and $v$ parameters. Furthermore, the script also computes the SVM solutions for both parameters as well. Results saved in `result/joint_convergence/{p}.pt`

The above bash scripts presents the results in the result directory, which can be visualized in the `plotter.ipynb` file.
