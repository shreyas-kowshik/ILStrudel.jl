using Pkg
Pkg.activate("/media/shreyas/Data/UCLA-Intern/ILStrudel/ILStrudel.jl")

using Test
using LogicCircuits
using ProbabilisticCircuits
using ILStrudel
using Statistics
using ArgParse
using JLD
using Random

function single_model(dataset; maxiter=100)
    # pc = learn_single_model("nltcs")
    train_x, valid_x, test_x = twenty_datasets(dataset)
    pick_edge = "eFlow"
    pick_var = "v_pMI"
    
    pc = learn_single_model(train_x, valid_x, test_x;
        pick_edge=pick_edge, pick_var=pick_var, depth=1,
        pseudocount=1.0,
        sanity_check=true,
        maxiter=maxiter,
        return_vtree=false)
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--name"
            help = "Name of the dataset"
            arg_type = String
            required = true
        
        "--maxiter"
            help = ""
            arg_type = Int
            default = 200
            required = false

        "--seed"
            help = ""
            arg_type = Int
            default = 42
            required = false
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

Random.seed!(parsed_args["seed"])
single_model(parsed_args["name"]; maxiter=parsed_args["maxiter"])
