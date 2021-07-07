# ILStrudel

Independence-Based Learning of Structured-Decomposable Circuits.

## Instructions To Run

Train on all datasets :

`python train.py`

`CUDA_VISIBLE_DEVICES=0 julia1 example_train.jl --name [dataset_name] --run_name [exp_name] --pseudocount 1.0 --maxiter 300 --pmi_thresh 0.03 --population_size 300 --num_mine_samples 7 --mine_iterations 3 --num_mi_bags 20 --seed 63`

To generate summary :

`julia1 example_summary.jl --logdir /space/shreyas-kowshik/runs/final10_42`
