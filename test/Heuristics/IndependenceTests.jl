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
    
end
