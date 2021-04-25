import os
import shlex, subprocess

# datasets = sorted(os.listdir('data'))
paper_data = ['accidents', 'ad', 'baudio', 'bbc', 'bnetflix', 'book', 'c20ng', 'cr52', 'cwebkb', 'dna', 'jester', 'kdd', 'kosarek', 'msnbc', 'msweb', 'nltcs', 'plants', 'pumsb_star', 'tmovie', 'tretail']

# datasets = sorted([d for d in datasets if d in paper_data])
datasets = paper_data

min_len = 0
max_len = 20
datasets = datasets[min_len:max_len]
# datasets = datasets[max_len:]
datasets = ['c20ng']

for (i,dataset) in enumerate(datasets):
    sess = dataset
    run_name = "mine_em_bagging_size_thresh_2"
    # command = "tmux new-session -d -s {} 'julia1 example_train.jl --name {} --run_name {} --pseudocount 1.0 --maxiter 150 --pmi_thresh 0.03 --population_size 300 --num_mine_samples 5 --mine_iterations 3 --num_mi_bags 50'".format(sess, dataset, run_name)
    command = "tmux new-session -d -s {} 'julia1 example_train.jl --name {} --run_name {} --pseudocount 1.0 --maxiter 1000 --pmi_thresh 0.03 --population_size 300 --num_mine_samples 5 --mine_iterations 3 --num_mi_bags 50 --bitmask_path ~/runs/mine_em_bagging_size_thresh_2/{}/bitmasks.jld'".format(sess, dataset, run_name, dataset)
    args = shlex.split(command)
    subprocess.Popen(args)

print(len(datasets))
