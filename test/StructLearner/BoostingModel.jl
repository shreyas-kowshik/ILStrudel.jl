using Pkg
Pkg.activate("/media/shreyas/Data/UCLA-Intern/ILStrudel/ILStrudel.jl")

using Test
using LogicCircuits
using ProbabilisticCircuits
using ILStrudel

@testset "Likelihood Computation" begin
    train_x, valid_x, test_x = twenty_datasets("nltcs")
    mixture = Mixture()
    pc, vtree = learn_chow_liu_tree_circuit(train_x)
    add_component(mixture, pc, 1.0)

    out = likelihood_per_instance(mixture, train_x)
    pc, vtree = learn_chow_liu_tree_circuit(train_x)
    add_component(mixture, pc, 0.5)
    out = likelihood_per_instance(mixture, train_x)
end

@testset "Direct Call" begin
    # pc = learn_single_model("nltcs")

    train_x, valid_x, test_x = twenty_datasets("nltcs")
    num_components = 5
    mixture = boosting(train_x, valid_x, test_x, num_components)
end