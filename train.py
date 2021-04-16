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

for (i,dataset) in enumerate(datasets):
    sess = dataset
    command = "tmux new-session -d -s {} 'julia example_train.jl --name {} --pseudocount 1.0 --maxiter 450 --pmi_thresh 0.03'".format(sess, dataset)
    args = shlex.split(command)
    subprocess.Popen(args)

print(len(datasets))
