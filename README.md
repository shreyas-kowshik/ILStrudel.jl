# ILStrudel

Code for the paper : <a href="https://openreview.net/forum?id=wAcMi8m5IIC">ILStrudel : Independence-Based Learning of Structured-Decomposable Probabilistic Circuit Ensembles</a> accepted at the <a href="https://sites.google.com/view/tpm2021/home">Tractable Probabilistic Models Workshop, UAI'21</a>.

## Instructions To Run

By default the code uses the GPU to speed-up pairwise-mutual-information computation.

Train on single dataset :

```
julia example_train.jl --name [dataset_name] --run_name [exp_name] --pseudocount 1.0 --maxiter 300 --pmi_thresh 0.03 --population_size 300 --num_mine_samples 7 --mine_iterations 3 --num_mi_bags 20 --seed 63
```

This will write all outputs to `$HOME_DIR/runs/$run_name/$dataset_name`.

To change the `$HOME_DIR` path, change the `$LOG_DIR` variable in `line 62` in `example_train.jl` to a custom path. 

To generate summary :

`julia example_summary.jl --logdir $HOME_DIR/runs/$run_name`

This will generate two files `runs.csv` and `summary.csv` under `$HOME_DIR/runs/$run_name`.

`runs.csv` will hold performance of all the different hyperparameter configurations for each dataset.

`summary.csv` will hold the best performing result of all the runs for each dataset.

Train on all datasets :

`python train.py`

This will launch 20 different tmux shells running in the background on each of the 20 datasets.

Change the parameters inside `train.py` according to the required experiment names and hyperparameters.

