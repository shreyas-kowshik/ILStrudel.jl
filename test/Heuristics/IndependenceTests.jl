using Pkg
Pkg.activate("/media/shreyas/Data/UCLA-Intern/ILStrudel/ILStrudel.jl")

using Test
using LogicCircuits
using ProbabilisticCircuits
using ILStrudel

@testset "Mutual-Information Calls" begin
    train_x, valid_x, test_x =  twenty_datasets("nltcs")
    # pc, vtree = learn_chow_liu_tree_circuit(train_x)
    dmat = BitArray(convert(Matrix, train_x))

    prime_vars = [1]
    sub_vars = [3]
    pmi_gpu = _mutual_information(dmat, prime_vars, sub_vars; use_gpu=true)
    pmi_cpu = _mutual_information(dmat, prime_vars, sub_vars; use_gpu=false)
    two_way_mi = _mutual_information(dmat, prime_vars, sub_vars;use_gpu=false, k=1)

    println("pmi_gpu : $pmi_gpu")
    println("pmi_cpu : $pmi_cpu")
    println("two_way_mi : $two_way_mi")
    println("---")

    prime_vars = [1, 5, 7, 9, 11]
    sub_vars = [2, 4, 6, 8, 10]
    pmi_gpu = _mutual_information(dmat, prime_vars, sub_vars; use_gpu=true)
    pmi_cpu = _mutual_information(dmat, prime_vars, sub_vars; use_gpu=false)
    two_way_mi = _mutual_information(dmat, prime_vars, sub_vars;use_gpu=false, k=1)

    println("pmi_gpu : $pmi_gpu")
    println("pmi_cpu : $pmi_cpu")
    println("two_way_mi : $two_way_mi")
    println("---")

    prime_vars = [2, 4, 6, 8, 10]
    sub_vars = [1, 5, 7, 9, 11]
    pmi_gpu = _mutual_information(dmat, prime_vars, sub_vars; use_gpu=true)
    pmi_cpu = _mutual_information(dmat, prime_vars, sub_vars; use_gpu=false)
    two_way_mi = _mutual_information(dmat, prime_vars, sub_vars;use_gpu=false, k=1)

    println("pmi_gpu : $pmi_gpu")
    println("pmi_cpu : $pmi_cpu")
    println("two_way_mi : $two_way_mi")
    println("---")
end

@testset "Mutual Information Values" begin
    println("Testing across bitmasks for pmi values")
    for dataset in twenty_dataset_names[1:end-1]
        println("Dataset : $dataset")
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
        for i in 1:10
            bm = BitArray(rand(d, n))
            pmi_cpu = _mutual_information(dmat[bm, :], prime_vars, sub_vars; use_gpu=false)
            pmi_gpu = _mutual_information(dmat[bm, :], prime_vars, sub_vars; use_gpu=true)

            @assert abs(pmi_cpu - pmi_gpu) < 1e-6 "Values on cpu : $pmi_cpu and gpu : $pmi_gpu not matching"
        end
    end
end
