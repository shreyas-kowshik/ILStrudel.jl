using Pkg
Pkg.activate("/media/shreyas/Data/UCLA-Intern/ILStrudel/ILStrudel.jl")

using Test
using LogicCircuits
using ProbabilisticCircuits
using ILStrudel
using Statistics
using ArgParse
using JLD

"""
Logs :
Save Bitmasks (some directory)
    - 
"""

BASE = homedir()
BITMASK_DIR = joinpath(BASE, "ILStrudel/bitmasks")
LOG_DIR = joinpath(BASE, "ILStrudel/log/boosting")
BAGGING_LOG_DIR = joinpath(BASE, "ILStrudel/log/bagging")

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

function save_bitmasks(path::String, bm_config)

end

function mine_model(dataset_name, config_dict; 
    mine_iterations=1,
    population_size=population_size,
    num_mine_samples=10,
    pseudocount=1e-9,
    sanity_check=true,
    maxiter=700,
    seed=nothing,
    return_vtree=false,
    return_bitmasks=true,
    pmi_thresh=0.1,
    load_bitmask_path=nothing,
    load_bitmasks=false)

    train_x, valid_x, test_x = twenty_datasets(dataset_name)
    pick_edge = "eFlow"
    pick_var = "vMI"
    config_name = "$(mine_iterations)_$(population_size)_$(num_mine_samples).jld"

    if !isnothing(load_bitmask_path)
        bitmasks = load(load_bitmask_path)["bitmasks"]
        println("Loaded Bitmasks!")
    else
        if load_bitmasks
            save_path = joinpath(BITMASK_DIR, dataset_name)
            load_bitmask_path = joinpath(save_path, config_name)
            bitmasks = load(load_bitmask_path)["bitmasks"]
            println("Loaded Bitmasks!")
        else
            bitmasks = nothing
        end
    end
    
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
        pmi_thresh=pmi_thresh,
        bitmasks=bitmasks)


    # Save Bitmasks
    save_path = joinpath(BITMASK_DIR, dataset_name)
    if !isdir(save_path)
        mkpath(save_path)
    end

    save_file = joinpath(save_path, config_name)
    save(save_file, "bitmasks", bitmasks)

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
    # idx = [t[2] for t in argmax(train_lls, dims=2)]
    # println(maximum(idx))
    vals = maximum(train_lls, dims=2)
    # vals = mean(train_lls, dims=2)
    train_ll = mean(vals)

    # idx = [t[2] for t in argmax(valid_lls, dims=2)]
    # println(maximum(idx))
    vals = maximum(train_lls, dims=2)
    # vals = mean(train_lls, dims=2)
    valid_ll = mean(vals)

    # idx = [t[2] for t in argmax(test_lls, dims=2)]
    # println(maximum(idx))
    vals = maximum(train_lls, dims=2)
    # vals = mean(train_lls, dims=2)
    test_ll = mean(vals)

    config_dict["train_ll"] = train_ll
    config_dict["valid_ll"] = valid_ll
    config_dict["test_ll"] = test_ll
    bit_lengths = [sum(b) for b in bitmasks]
    total_params = sum([num_parameters(pc) for pc in pcs])
    config_dict["params"] = total_params

    # Save Results
    save_path = joinpath(LOG_DIR, dataset_name)
    if !isdir(save_path)
        mkpath(save_path)
    end

    file_id = length(readdir(save_path)) + 1
    file_name = "$(file_id).jld"
    save_file = joinpath(save_path, file_name)
    save(save_file, "config_dict", config_dict)

    println(config_dict)
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

function boosting_model(dataset_name, config_dict; maxiter=100, pseudocount=1.0, num_boosting_components=5,
                        share_structure=false)
    train_x, valid_x, test_x = twenty_datasets(dataset_name)

    if share_structure
        mixture = boosting_shared_structure(train_x, valid_x, test_x, num_boosting_components; pseudocount=pseudocount, maxiter=maxiter)
    else
        mixture = boosting(train_x, valid_x, test_x, num_boosting_components; pseudocount=pseudocount, maxiter=maxiter)
    end
    train_ll = mean(mixture_log_likelihood_per_instance(mixture, train_x))
    valid_ll = mean(mixture_log_likelihood_per_instance(mixture, valid_x))
    test_ll = mean(mixture_log_likelihood_per_instance(mixture, test_x))
    num_params = sum([num_parameters(pc) for pc in mixture.components])

    config_dict["train_ll"] = train_ll
    config_dict["valid_ll"] = valid_ll
    config_dict["test_ll"] = test_ll
    config_dict["params"] = num_params

    save_path = joinpath(LOG_DIR, dataset_name)
    if !isdir(save_path)
        mkpath(save_path)
    end
    
    file_id = length(readdir(save_path)) + 1
    file_name = "boosting_$(file_id).jld"
    save_file = joinpath(save_path, file_name)
    save(save_file, "config_dict", config_dict)

    println(config_dict)
end

function bagging_em_model(dataset_name, config_dict; maxiter=100, pseudocount=1.0, num_bagging_components=5,
                        num_em_components=3)
    train_x, valid_x, test_x = twenty_datasets(dataset_name)

    bags = bagging(train_x, valid_x, test_x, num_bagging_components; num_em_components=num_em_components,
		   maxiter=maxiter, pseudocount=pseudocount)
    train_lls = hcat([mixture_log_likelihood_per_instance(mixture, train_x) for mixture in bags]...)
    valid_lls = hcat([mixture_log_likelihood_per_instance(mixture, valid_x) for mixture in bags]...)
    test_lls = hcat([mixture_log_likelihood_per_instance(mixture, test_x) for mixture in bags]...)

    println("\n\n\nSize : $(size(train_lls))\n\n\n")
    train_ll = mean(logsumexp(train_lls, dims=2) .+ log.(1.0 ./ num_bagging_components))
    valid_ll = mean(logsumexp(valid_lls, dims=2) .+ log.(1.0 ./ num_bagging_components))
    test_ll = mean(logsumexp(test_lls, dims=2) .+ log.(1.0 ./ num_bagging_components))

    num_params = sum([sum([num_parameters(pc) for pc in mixture.components]) for mixture in bags])

    config_dict["train_ll"] = train_ll
    config_dict["valid_ll"] = valid_ll
    config_dict["test_ll"] = test_ll
    config_dict["params"] = num_params

    save_path = joinpath(BAGGING_LOG_DIR, dataset_name)
    if !isdir(save_path)
        mkpath(save_path)
    end
    
    file_id = length(readdir(save_path)) + 1
    file_name = "bagging_$(file_id).jld"
    save_file = joinpath(save_path, file_name)
    save(save_file, "config_dict", config_dict)

    println(config_dict)
end

function weighted_single_model(dataset_name, config_dict; pseudocount=1.0, maxiter=200)
    train_x, valid_x, test_x = twenty_datasets(dataset_name)
    N = size(train_x)[1]
    w = ones(N)
    w[1:100] = 2.0
    w[101:200] = 5.0
    w = (w .- minimum(w)) ./ (maximum(w) - minimum(w))
    println(sum(w))

    weighted_train_x = add_sample_weights(train_x, w)
    pc, vtree = learn_weighted_chow_liu_tree_circuit(weighted_train_x)
    pc = learn_single_model(weighted_train_x, valid_x, test_x, pc, vtree;
                           pseudocount=pseudocount,
                           maxiter=maxiter)

    train_lls = mean(log_likelihood_per_instance(pc, train_x))
    valid_lls = mean(log_likelihood_per_instance(pc, valid_x))
    test_lls = mean(log_likelihood_per_instance(pc, test_x))
    config_dict["train_ll"] = train_ll
    config_dict["valid_ll"] = valid_ll
    config_dict["test_ll"] = test_ll
    println(config_dict)
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
        
        "--bitmask_path"
            help = "Path where bitmasks are saved"
            arg_type = String
            default = nothing
            required = false
        
        "--load_bitmasks"
            help = "Whether to load bitmasks"
            arg_type = Bool
            default = false
            required = false
        
        "--num_boosting_components"
            help = "Number of Boosting Components"
            arg_type = Int
            default = 5
            required = false

        "--num_bagging_components"
            help = "Number of Boosting Components"
            arg_type = Int
            default = 5
            required = false
	
        "--num_em_components"
            help = "Number of Boosting Components"
            arg_type = Int
            default = 5
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

mine_model(parsed_args["name"], parsed_args;
mine_iterations=parsed_args["mine_iterations"],
population_size=parsed_args["population_size"],
num_mine_samples=parsed_args["num_mine_samples"],
pseudocount=parsed_args["pseudocount"],
maxiter=parsed_args["maxiter"],
pmi_thresh=parsed_args["pmi_thresh"],
load_bitmask_path=parsed_args["bitmask_path"],
load_bitmasks=parsed_args["load_bitmasks"])

# boosting_model(parsed_args["name"], parsed_args;
#               maxiter=parsed_args["maxiter"],
#               pseudocount=parsed_args["pseudocount"],
#               num_boosting_components=parsed_args["num_boosting_components"])

# bagging_em_model(parsed_args["name"], parsed_args;
#               maxiter=parsed_args["maxiter"],
#               pseudocount=parsed_args["pseudocount"],
#               num_bagging_components=parsed_args["num_bagging_components"],
# 	      num_em_components=parsed_args["num_em_components"])

# weighted_single_model(parsed_args["name"], parsed_args;
#               maxiter=parsed_args["maxiter"],
#               pseudocount=parsed_args["pseudocount"])

