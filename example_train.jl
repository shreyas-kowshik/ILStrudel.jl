using Pkg
Pkg.activate("/media/shreyas/Data/UCLA-Intern/ILStrudel/ILStrudel.jl")

using Test
using LogicCircuits
using ProbabilisticCircuits
using ILStrudel
using Statistics
using ArgParse

"""
(depth=3)
julia example_train.jl --name dna --population_size 50000 
--pmi_thresh 0.01 --pseudocount 1.0 --maxiter 200 --num_mine_samples 10 --mine_iterations 3
"""

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

function mine_model(dataset_name; 
    mine_iterations=1,
    population_size=population_size,
    num_mine_samples=10,
    pseudocount=1e-9,
    sanity_check=true,
    maxiter=700,
    seed=nothing,
    return_vtree=false,
    return_bitmasks=true,
    pmi_thresh=0.1)

    train_x, valid_x, test_x = twenty_datasets(dataset_name)
    pick_edge = "eFlow"
    pick_var = "vMI"
    
    pcs, bitmasks = learn_mine_ensemble(train_x, valid_x, test_x;
        mine_iterations=mine_iterations,
        population_size=population_size,
        num_mine_samples=num_mine_samples,
        pick_edge=pick_edge, pick_var=pick_var, depth=3,
        pseudocount=pseudocount,
        sanity_check=sanity_check,
        maxiter=maxiter,
        seed=seed,
        return_vtree=return_vtree,
        return_bitmasks=return_bitmasks,
        pmi_thresh=pmi_thresh)

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
    bit_lengths = [length(b) for b in bitmasks]
    total_params = sum([num_parameters(pc) for pc in pcs])
    println(bit_lengths)
    println("Total Parameters : $(total_params)")
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
        
        "--mine_iterations"
            help = "Number of iterations to mine data"
            arg_type = Int
            default = 1
            required = false

        "--population_size"
            help = ""
            arg_type = Int
            default = 1000
            required = false
            
        "--num_mine_samples"
            help = ""
            arg_type = Int
            default = 10
      		required = false
        
        "--pseudocount"
            help = ""
            arg_type = Float64
            default = 1e-9
            required = false
        
        "--maxiter"
            help = ""
            arg_type = Int
            default = 200
            required = false
              
        "--pmi_thresh"
            help = ""
            arg_type = Float64
            default = 0.1
      		required = false

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

mine_model(parsed_args["name"];
mine_iterations=parsed_args["mine_iterations"],
population_size=parsed_args["population_size"],
num_mine_samples=parsed_args["num_mine_samples"],
pseudocount=parsed_args["pseudocount"],
maxiter=parsed_args["maxiter"],
pmi_thresh=parsed_args["pmi_thresh"])
