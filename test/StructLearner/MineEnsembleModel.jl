using Pkg
Pkg.activate("/media/shreyas/Data/UCLA-Intern/ILStrudel/ILStrudel.jl")

using Test
using LogicCircuits
using ProbabilisticCircuits
using ILStrudel

@testset "Direct Call" begin
    # pc = learn_single_model("nltcs")

    train_x, valid_x, test_x = twenty_datasets("nltcs")
    pick_edge = "eFlow"
    pick_var = "vMI"
    
    pcs = learn_mine_ensemble(train_x, valid_x, test_x;
        mine_iterations=3,
        num_mine_samples=10,
        pick_edge=pick_edge, pick_var=pick_var, depth=1, 
        pseudocount=1.0,
        sanity_check=true,
        maxiter=100,
        seed=nothing,
        return_vtree=false)
end
