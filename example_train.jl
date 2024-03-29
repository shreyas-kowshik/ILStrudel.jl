using Pkg
Pkg.activate(".") # Change according to the absolute path if required

using Test
using LogicCircuits
using ProbabilisticCircuits
using ILStrudel
using Statistics
using ArgParse
using JLD
using Random

"""
Circuit I/O template :
save_circuit("tem.psdd", pc, vtree)
save_vtree("tem.vtree", vtree)
pc1 = load_struct_prob_circuit("tem.psdd", "tem.vtree")
"""

BASE = homedir()
BITMASK_DIR = joinpath(BASE, "ILStrudel/bitmasks_bag_mi")
LOG_DIR = joinpath(BASE, "ILStrudel/log/mine_bag_mi")
BAGGING_LOG_DIR = joinpath(BASE, "ILStrudel/log/bagging_mine_bag_mi")

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

function mine_em_model(dataset_name, config_dict; 
    mine_iterations=1,
    population_size=population_size,
    num_mine_samples=10,
    pseudocount=1e-9,
    sanity_check=true,
    maxiter=700,
    return_vtree=false,
    return_bitmasks=true,
    pmi_thresh=0.1,
    load_bitmask_path=nothing,
    load_bitmasks=false,
    seed=42)

    Random.seed!(seed)
    train_x, valid_x, test_x = twenty_datasets(dataset_name)
    pick_edge = "eFlow"
    pick_var = "vMI"
    config_name = "$(mine_iterations)_$(population_size)_$(num_mine_samples).jld"
    base_pc = nothing
    base_vtree = nothing

    LOG_DIR = joinpath(BASE, "runs/", config_dict["run_name"])
    if !isdir(LOG_DIR)
    	mkpath(LOG_DIR)
    end

    if !isnothing(load_bitmask_path)
        println("Loading Bitmasks from path : $load_bitmask_path")
        bitmasks = load(load_bitmask_path)["bitmasks"]
        println("Loaded Bitmasks!")
        load_pc_path = joinpath(load_bitmask_path[1:end-12], "base_pc.psdd")
        load_vtree_path = joinpath(load_bitmask_path[1:end-12], "base_vtree.vtree")
        println("Loading Base PC from path : $load_pc_path")
        base_pc, base_vtree = load_struct_prob_circuit(load_pc_path, load_vtree_path)
    else
        if load_bitmasks
	    error("This part is not to be used now")
            save_path = joinpath(BITMASK_DIR, dataset_name)
            load_bitmask_path = joinpath(save_path, config_name)
            bitmasks = load(load_bitmask_path)["bitmasks"]
            println("Loaded Bitmasks!")
        else
            bitmasks = nothing
        end
    end
    
    # Get pmi_thresh from files
    # Mean of values chosen
    pmi_stats_path = joinpath("bin/pmi_unique_stats_47/", dataset_name)
    pmi_stats = collect(values(load(joinpath(pmi_stats_path, string("mi", config_dict["num_mi_bags"], "s.jld")))))[1]
    pmi_stats = pmi_stats[401:500]
    println("Length pmi_stats : $(length(pmi_stats))")
    # Set mean + std of the statistics
    pmi_thresh = mean(pmi_stats)

    println("PMI THRESH USED : $pmi_thresh")

    # Get Size Threshold
    N = size(train_x)[1]
    size_thresh = floor(Int, N / num_mine_samples)
    println("SIZE THRESH USED : $size_thresh")

    pcs, bitmasks, pmis, base_pc, base_vtree = learn_mine_ensemble(train_x, valid_x, test_x;
        mine_iterations=mine_iterations,
        population_size=population_size,
        num_mine_samples=num_mine_samples,
        pick_edge=pick_edge, pick_var=pick_var, depth=config_dict["depth"],
        pseudocount=pseudocount,
        sanity_check=sanity_check,
        maxiter=maxiter,
        seed=seed,
        return_vtree=return_vtree,
        return_bitmasks=return_bitmasks,
        pmi_thresh=pmi_thresh,
        size_thresh=size_thresh,
        bitmasks=bitmasks,
        pc=deepcopy(base_pc),
        vtree=deepcopy(base_vtree))

    # Save the bitmasks
    bitmask_save_path = joinpath(LOG_DIR, dataset_name)
    if !isdir(bitmask_save_path)
        mkpath(bitmask_save_path)
    end

    pc_save_path = joinpath(bitmask_save_path, "base_pc.psdd")
    vtree_save_path = joinpath(bitmask_save_path, "base_vtree.vtree")
    bitmask_save_path = joinpath(bitmask_save_path, "bitmasks.jld")
    save_circuit(pc_save_path, base_pc, base_pc.vtree)
    save_vtree(vtree_save_path, base_pc.vtree)
    save(bitmask_save_path, "bitmasks", bitmasks)

    mixture = Mixture()
    for pc in pcs
        # TODO : Experiment with this
        # Re-estimate parameters over the entire dataset as a starting point
        estimate_parameters(pc, train_x; pseudocount=pseudocount)
        add_component(mixture, pc)
    end

    # Get initial vtree
    # _, vtree = learn_chow_liu_tree_circuit(train_x)

    weights = [sum(bm) / length(bm) for bm in bitmasks]
    weights = weights ./ sum(weights)
    mixture, data_weights = EM(mixture, train_x; weights=weights, pseudocount=pseudocount)
    train_ll = mean(mixture_log_likelihood_per_instance(mixture, train_x))
    valid_ll = mean(mixture_log_likelihood_per_instance(mixture, valid_x))
    test_ll = mean(mixture_log_likelihood_per_instance(mixture, test_x))
    num_params = sum([num_parameters(pc) for pc in mixture.components])
    println("Mixture Weights : $(mixture.weights)")

    config_dict["train_ll"] = train_ll
    config_dict["valid_ll"] = valid_ll
    config_dict["test_ll"] = test_ll
    config_dict["params"] = num_params
    config_dict["pmis"] = pmis
    config_dict["em_data_weights"] = data_weights
    config_dict["pmi_thresh"] = pmi_thresh
    config_dict["size_thresh"] = size_thresh

    save_path = joinpath(LOG_DIR, dataset_name)
    if !isdir(save_path)
        mkpath(save_path)
    end
    
    config_dict["em_weights"] = mixture.weights
    file_id = length(readdir(save_path)) + 1
    file_name = "mine_em_$(file_id).jld"
    save_file = joinpath(save_path, file_name)
    save(save_file, "config_dict", config_dict)

    weights_path = joinpath(save_path, "mine_em_$(file_id)")
    if !isdir(weights_path)
        mkpath(weights_path)
    end
    for (i, pc) in enumerate(mixture.components)
        save_circuit(joinpath(weights_path, "pc_$i.psdd"), pc, pc.vtree)
	save_vtree(joinpath(weights_path, "vt_$i.vtree"), pc.vtree)
    end

    println(train_ll)
    println(valid_ll)
    println("Test LL : $test_ll")
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

        "--run_name"
            help = "Name of the run"
            arg_type = String
            required = true
        
        "--mine_iterations"
            help = "Number of iterations to mine data"
            arg_type = Int
            default = 1
            required = false

        "--population_size"
            help = "Initial population size for the genetic algorithm"
            arg_type = Int
            default = 1000
            required = false
            
        "--num_mine_samples"
            help = "Number of partitions to divide the data into"
            arg_type = Int
            default = 10
      		required = false
        
        "--pseudocount"
            help = "Pseudocount value for parameter estimation"
            arg_type = Float64
            default = 1.0
            required = false
        
        "--maxiter"
            help = "Maximum number of iterations to run structure-learner for each component"
            arg_type = Int
            default = 200
            required = false

        "--depth"
            help = ""
            arg_type = Int
            default = 1
            required = false
              
        "--pmi_thresh"
            help = "This is a redundant parameter. Not used per-se. Thresholds are chosen according to simulated statistics."
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

        "--num_mi_bags"
            help = "Number of Bootstraps for mutual information computation"
            arg_type = Int
            default = 10
            required = false
       
        "--seed"
            help = "Random Seed"
            arg_type = Int
            default = 42
            required = false
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

mine_em_model(parsed_args["name"], parsed_args;
mine_iterations=parsed_args["mine_iterations"],
population_size=parsed_args["population_size"],
num_mine_samples=parsed_args["num_mine_samples"],
pseudocount=parsed_args["pseudocount"],
maxiter=parsed_args["maxiter"],
pmi_thresh=parsed_args["pmi_thresh"],
load_bitmask_path=parsed_args["bitmask_path"],
load_bitmasks=parsed_args["load_bitmasks"],
seed=parsed_args["seed"])

# boosting_model(parsed_args["name"], parsed_args;
#              maxiter=parsed_args["maxiter"],
#              pseudocount=parsed_args["pseudocount"],
#              num_boosting_components=parsed_args["num_boosting_components"])

# bagging_em_model(parsed_args["name"], parsed_args;
#               maxiter=parsed_args["maxiter"],
#               pseudocount=parsed_args["pseudocount"],
#               num_bagging_components=parsed_args["num_bagging_components"],
# 	      num_em_components=parsed_args["num_em_components"])

# weighted_single_model(parsed_args["name"], parsed_args;
#               maxiter=parsed_args["maxiter"],
#               pseudocount=parsed_args["pseudocount"])

