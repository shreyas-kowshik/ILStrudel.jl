using Distributions
using PyPlot
using JLD

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