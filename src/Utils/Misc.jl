using Distributions
using DataFrames
using PyPlot
using JLD
using CSV

function generate_pmi_bagging_stats(dataset; num_iters=1000)
    train_x, _, _ = twenty_datasets(dataset)
    pc, vtree = learn_chow_liu_tree_circuit(train_x)

    dmat = BitArray(convert(Matrix, train_x))

    and = children(pc)[1]
    og_lits = collect(Set{Lit}(variables(and.vtree))) # All literals

    prime_lits = sort([abs(l) for l in og_lits if l in variables(children(and)[1].vtree)])
    sub_lits = sort([abs(l) for l in og_lits if l in variables(children(and)[2].vtree)])

    prime_lits = sort(collect(Set{Lit}(prime_lits)))
    sub_lits = sort(collect(Set{Lit}(sub_lits)))
    prime_sub_lits = sort([prime_lits..., sub_lits...])

    n = size(dmat)[1]
    p= 0.5
    d = Binomial(1, p)

    mis = []
    mi20s = []
    mi50s = []
    for i in 1:num_iters
    	println("Iteration : $i")
        bm = BitArray(rand(d, n))
        mi = bootstrap_mutual_information(dmat[bm, :], prime_lits, sub_lits; num_bags=1, use_gpu=false, k=1, α=1.0)
        mi20 = bootstrap_mutual_information(dmat[bm, :], prime_lits, sub_lits; num_bags=20, use_gpu=false, k=1, α=1.0)
        mi50 = bootstrap_mutual_information(dmat[bm, :], prime_lits, sub_lits; num_bags=50, use_gpu=false, k=1, α=1.0)

        push!(mis, mi)
        push!(mi20s, mi20)
        push!(mi50s, mi50)
    end

    if !isdir(joinpath("bin/pmi_stats", dataset))
        mkpath(joinpath("bin/pmi_stats", dataset))
    end
    
    save_file = joinpath("bin/pmi_stats", dataset, "mis.jld")
    save(save_file, "mis", mis)
    save_file = joinpath("bin/pmi_stats", dataset, "mi20s.jld")
    save(save_file, "mi20s", mi20s)
    save_file = joinpath("bin/pmi_stats", dataset, "mi50s.jld")
    save(save_file, "mi50s", mi50s)

    hist(mis, bins=20)
    savefig(joinpath("bin/pmi_stats", dataset, "mis.png"))
    hist(mi20s, bins=20)
    savefig(joinpath("bin/pmi_stats", dataset, "mi20s.png"))
    hist(mi50s, bins=20)
    savefig(joinpath("bin/pmi_stats", dataset, "mi50s.png"))

end

"""
dir : directory containing dataset names and `.jld` files
"""
function generate_plots(dir; bins=50)
    ioff()
    ld(x) = collect(values(load(x)))[1]

    for dataset in readdir(dir)
        path = joinpath(dir, dataset)
        mis = ld(joinpath(path, "mis.jld"))
        mi20s = ld(joinpath(path, "mi20s.jld"))
        mi50s = ld(joinpath(path, "mi50s.jld"))

        fig, ax = subplots(3)
        fig.suptitle(dataset)
        ax[1].hist(mis, bins=bins)
        ax[2].hist(mis, bins=bins)
        ax[2].hist(mi20s, bins=bins)
        ax[3].hist(mis, bins=bins)
        ax[3].hist(mi20s, bins=bins)
        ax[3].hist(mi50s, bins=bins)
        savefig(joinpath(path, "stats.png"))
    end
end

function save_as_csv(dict; filename, header=keys(dict))
    table = DataFrame(;[Symbol(x) => dict[x] for x in header]...)
    CSV.write(filename, table; )
end

function generate_pmi_runtime_stats()
	save_dir = joinpath("bin")
	if !isdir(save_dir)
		mkpath(save_dir)
	end

	datasets = []
	mis_cpu = []
	mi20s_cpu = []
	mis_gpu = []
	mi20s_gpu = []


	for dataset in twenty_dataset_names[1:end-1]
		println(dataset)
		train_x, _, _ = twenty_datasets(dataset)
		println(size(train_x))
	    	pc, vtree = learn_chow_liu_tree_circuit(train_x)
		dmat = BitArray(convert(Matrix, train_x))

    		and = children(pc)[1]
    		og_lits = collect(Set{Lit}(variables(and.vtree))) # All literals

	    	prime_lits = sort([abs(l) for l in og_lits if l in variables(children(and)[1].vtree)])
    		sub_lits = sort([abs(l) for l in og_lits if l in variables(children(and)[2].vtree)])

    		prime_lits = sort(collect(Set{Lit}(prime_lits)))
    		sub_lits = sort(collect(Set{Lit}(sub_lits)))
    		prime_sub_lits = sort([prime_lits..., sub_lits...])


		# The first gpu kernel is slower, so to compensate for that
    		t = bootstrap_mutual_information(dmat, prime_lits, sub_lits; num_bags=1, use_gpu=true, k=1, α=1.0)

		mi_cpu_val = _mutual_information(dmat, prime_lits, sub_lits; k=1, use_gpu=false, α=1.0)
		mi_gpu_val = _mutual_information(dmat, prime_lits, sub_lits; k=1, use_gpu=true, α=1.0)
		
		t0 = Base.time_ns()
    		mi_cpu = bootstrap_mutual_information(dmat, prime_lits, sub_lits; num_bags=1, use_gpu=false, k=1, α=1.0)
		t1 = Base.time_ns()
		mi_cpu = (t1 - t0)/1e9
		println(mi_cpu)
		
		t0 = Base.time_ns()
    		mi20_cpu = bootstrap_mutual_information(dmat, prime_lits, sub_lits; num_bags=20, use_gpu=false, k=1, α=1.0)
		t1 = Base.time_ns()
		mi20_cpu = (t1 - t0)/1e9
		println(mi20_cpu)
		
		t0 = Base.time_ns()
    		mi_gpu = bootstrap_mutual_information(dmat, prime_lits, sub_lits; num_bags=1, use_gpu=true, k=1, α=1.0)
		t1 = Base.time_ns()
		mi_gpu = (t1 - t0)/1e9
		println(mi_gpu)
	
		t0 = Base.time_ns()
    		mi20_gpu = bootstrap_mutual_information(dmat, prime_lits, sub_lits; num_bags=20, use_gpu=true, k=1, α=1.0)
		t1 = Base.time_ns()
		mi20_gpu = (t1 - t0)/1e9
		println(mi20_gpu)

		@assert isapprox(mi_cpu_val, mi_gpu_val; atol=1e-6) "cpu : $mi_cpu_val, gpu : $mi_gpu_val"

    		push!(datasets, dataset)
    		push!(mis_cpu, mi_cpu)
    		push!(mi20s_cpu, mi20_cpu)
    		push!(mis_gpu, mi_gpu)
    		push!(mi20s_gpu, mi20_gpu)
    	end

	summary_dict = Dict()
	summary_dict["dataset"] = datasets
	summary_dict["mis_cpu"] = mis_cpu
	summary_dict["mi20s_cpu"] = mi20s_cpu
	summary_dict["mis_gpu"] = mis_gpu
	summary_dict["mi20s_gpu"] = mi20s_gpu
	
	header = ["dataset", "mis_cpu", "mi20s_cpu", "mis_gpu", "mi20s_gpu"]
	save_as_csv(summary_dict; filename=joinpath(save_dir, "pmi_runtimes.csv"), header=header)
	println("Saved Runtimes")
end

function plot_instance_frequency()
	println("Plotting instance frequencies")
	for dataset in twenty_dataset_names[1:end-1]
		train_x, _, _ = twenty_datasets(dataset)
		println(size(train_x))
	    pc, vtree = learn_chow_liu_tree_circuit(train_x)

		# Unique mapping
		uq = unique(train_x)
		dict = Dict()
		for i in 1:size(uq)[1]
			dict[uq[i, :]] = []
		end
	
		for i in 1:size(train_x)[1]
			push!(dict[train_x[i, :]], i)
		end

		freq_arr = []
		for i in 1:size(uq)[1]
			push!(freq_arr, length(dict[uq[i, :]]))
		end
		idx = [i for i in 1:length(freq_arr)]

		if !isdir(joinpath("bin", "freqs", dataset))
			mkpath(joinpath("bin", "freqs", dataset))
		end

		path = joinpath("bin", "freqs", dataset)
		fig, ax = subplots(1)
        fig.suptitle(dataset)
        ax.bar(idx, freq_arr)
        savefig(joinpath(path, "freqs_distribution.png"))
	end
end