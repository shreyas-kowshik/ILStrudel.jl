using Distributions
using PyPlot

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
        bm = BitArray(rand(d, n))
        mi = bootstrap_mutual_information(dmat[bm, :], prime_lits, sub_lits; num_bags=1, use_gpu=false, k=1, α=1.0)
        mi20 = bootstrap_mutual_information(dmat[bm, :], prime_lits, sub_lits; num_bags=20, use_gpu=false, k=1, α=1.0)
        mi50 = bootstrap_mutual_information(dmat[bm, :], prime_lits, sub_lits; num_bags=1, use_gpu=false, k=1, α=1.0)

        push!(mis)
        push!(mi20s)
        push!(mi50s)
    end

    if !isdir(joinpath("bin/pmi_stats", dataset))
        mkpath(joinpath("bin/pmi_stats", dataset))
    end

    hist(mis, bins=20)
    savefig(joinpath("bin/pmi_stats", dataset, "mis.png"))
    hist(mi20s, bins=20)
    savefig(joinpath("bin/pmi_stats", dataset, "mi20s.png"))
    hist(mi50s, bins=20)
    savefig(joinpath("bin/pmi_stats", dataset, "mi50s.png"))

end
