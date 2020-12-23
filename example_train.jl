using Pkg
Pkg.activate("/media/shreyas/Data/UCLA-Intern/ILStrudel/ILStrudel.jl")

using Test
using LogicCircuits
using ProbabilisticCircuits
using ILStrudel
using Statistics

function single_model()
    # pc = learn_single_model("nltcs")
    train_x, valid_x, test_x = twenty_datasets("nltcs")
    pick_edge = "w_ind"
    pick_var = "w_ind"
    
    pc = learn_single_model(train_x, valid_x, test_x;
        pick_edge=pick_edge, pick_var=pick_var, depth=1,
        pseudocount=1.0,
        sanity_check=true,
        maxiter=5000,
        seed=nothing,
        return_vtree=false)
end

function mine_model()
    train_x, valid_x, test_x = twenty_datasets("plants")
    pick_edge = "eFlow"
    pick_var = "vMI"
    population_size=100
    
    pcs, bitmasks = learn_mine_ensemble(train_x, valid_x, test_x;
        mine_iterations=1,
        population_size=population_size,
        num_mine_samples=5,
        pick_edge=pick_edge, pick_var=pick_var, depth=1, 
        pseudocount=1e-9,
        sanity_check=true,
        maxiter=200,
        seed=nothing,
        return_vtree=false,
        return_bitmasks=true)

    # Validation ll computation
    weights = [sum(bitmask) / size(train_x)[1] for bitmask in bitmasks]
    println(weights)
    println(sum(weights))

    train_lls = hcat([log_likelihood_per_instance(pc, train_x) for pc in pcs]...)
    valid_lls = hcat([log_likelihood_per_instance(pc, valid_x) for pc in pcs]...)
    test_lls = hcat([log_likelihood_per_instance(pc, test_x) for pc in pcs]...)

    println(size(valid_lls))
    println(size(test_lls))

    ###
    idx = [t[2] for t in argmax(train_lls, dims=2)]
    println(maximum(idx))
    vals = maximum(train_lls, dims=2)
    train_ll = mean(vals)

    idx = [t[2] for t in argmax(valid_lls, dims=2)]
    println(maximum(idx))
    vals = maximum(valid_lls, dims=2)
    valid_ll = mean(vals)

    idx = [t[2] for t in argmax(test_lls, dims=2)]
    println(maximum(idx))
    vals = maximum(test_lls, dims=2)
    test_ll = mean(vals)

    println("Train LL : $(train_ll)")
    println("Valid_LL : $(valid_ll)")
    println("Test LL : $(test_ll)")
    ###


    # prod_nodes = []
    # for pc in pcs
    #     push!(prod_nodes, children(pc)...)
    # end

    # ensemble_pc = disjoin(prod_nodes...)
    # estimate_parameters(ensemble_pc, train_x; pseudocount=1.0)
    # valid_ll = log_likelihood_avg(ensemble_pc, valid_x)
    # test_ll = log_likelihood_avg(ensemble_pc, test_x)
    # println(valid_ll)
    # println(test_ll)
end

mine_model()
