using Pkg
Pkg.activate("/media/shreyas/Data/UCLA-Intern/ILStrudel/ILStrudel.jl")

using Test
using LogicCircuits
using ProbabilisticCircuits
using ILStrudel
using Statistics
using ArgParse

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

function mine_model(dataset_name)
    train_x, valid_x, test_x = twenty_datasets(dataset_name)
    pick_edge = "eFlow"
    pick_var = "vMI"
    population_size=1000
    
    pcs, bitmasks = learn_mine_ensemble(train_x, valid_x, test_x;
        mine_iterations=1,
        population_size=population_size,
        num_mine_samples=10,
        pick_edge=pick_edge, pick_var=pick_var, depth=1,
        pseudocount=1e-9,
        sanity_check=true,
        maxiter=350,
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

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--name"
            help = "Name of the dataset"
            arg_type = String
      		required = true

        # "--split_h"
        #     help = "Split Heuristic"
        #     arg_type = String
        #     required = true

        # "--clone_h"
        #     help = "Clone Heuristic"
        #     arg_type = String
        #     required = true

        # "--merge_h"
        #     help = "Merge Heuristic"
        #     arg_type = String
        #     required = true
        # "--opt2", "-o"
        #     help = "another option with an argument"
        #     arg_type = Int
        #     default = 0
        # "--flag1"
        #     help = "an option without argument, i.e. a flag"
        #     action = :store_true
        # "arg1"
        #     help = "a positional argument"
        #     required = true
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

mine_model(parsed_args["name"])