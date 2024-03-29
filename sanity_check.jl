using JLD
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--logdir"
            help = "Experiment Directory"
            arg_type = String
            required = false
            default = joinpath("/space/shreyas-kowshik/ILStrudel/log")
        "--dataset"
            help = "Dataset"
            arg_type = String
            required = false
            default = joinpath("/space/shreyas-kowshik/ILStrudel/log")
        "--run_name"
            help = "Experiment Run Name"
            arg_type = String
            required = false
            default = joinpath("/space/shreyas-kowshik/ILStrudel/log")
        "--exhaust"
            help = "Experiment Run Name"
            arg_type = Bool
            required = false
            default = false
    end
    return parse_args(s)
end


# function non_exhaust(parsed_args)
# dataset_dir = joinpath(parsed_args["logdir"], parsed_args["dataset"])
# pc_path = joinpath(dataset_dir, parsed_args["run_name"])
# config_dict = load(joinpath(dataset_dir, string(parsed_args["run_name"], ".jld")))["config_dict"]
# em_wts = config_dict["em_weights"]


# pcs = []
# for i in 1:Int(length(readdir(pc_path))/2)
#  pc1 = load_struct_prob_circuit(joinpath(pc_path,string("pc_", string(i), ".psdd")), joinpath(pc_path,string("vt_", string(i), ".vtree")))
#  push!(pcs, pc1)
# end

# pcs = first.(pcs)
# mixture = Mixture()
# for pc in pcs
#  add_component(mixture, pc)
# end

# mixture.weights = em_wts

# ll(x) = mixture_log_likelihood_per_instance(mixture, x);

# train_x, valid_x, test_x = twenty_datasets(parsed_args["dataset"]);
# uq = unique(train_x)

# train_ll = ll(uq)
# lse = logsumexp(train_ll)
# se = sum(exp.(train_ll))

# println(parsed_args["logdir"])
# println(parsed_args["dataset"])
# println(parsed_args["run_name"])
# println("Logsumexp : $lse")
# println("Sumexmp : $(se)")

# open(joinpath("bin", "sum_probs.txt"), "a") do file
#  write(file, parsed_args["logdir"])
#  write(file, "\n")
#  write(file, parsed_args["dataset"])
#  write(file, "\n")
#  write(file, parsed_args["run_name"])
#  write(file, "\n")
#  write(file, string("Logsumexp : ", string(lse)))
#  write(file, "\n")
#  write(file, string("Sumexp : ", string(se)))
#  write(file, "\n\n\n")
# end
# end


parsed_args = parse_commandline()

# function exhaust(parsed_args)
println("IN EXHAUST!")
dataset_dir = joinpath(parsed_args["logdir"], parsed_args["dataset"])
pc_path = joinpath(dataset_dir, parsed_args["run_name"])
config_dict = load(joinpath(dataset_dir, string(parsed_args["run_name"], ".jld")))["config_dict"]
em_wts = config_dict["em_weights"]

using LogicCircuits
using ProbabilisticCircuits
using ILStrudel
using DataFrames

pcs = []
for i in 1:Int(length(readdir(pc_path))/2)
 pc1 = load_struct_prob_circuit(joinpath(pc_path,string("pc_", string(i), ".psdd")), joinpath(pc_path,string("vt_", string(i), ".vtree")))
 push!(pcs, pc1)
end

pcs = first.(pcs)
mixture = Mixture()
for pc in pcs
 add_component(mixture, pc)
end

mixture.weights = em_wts

ll(x) = mixture_log_likelihood_per_instance(mixture, x);

train_x, valid_x, test_x = twenty_datasets(parsed_args["dataset"]);
# uq = unique(train_x)

# Build uq from exhaustive examples
num_vars = Int(size(train_x)[2])
instances = []
for i in 0:(2^num_vars - 1)
 # println(i)
 bs = bitstring(i)[end-num_vars+1:end]
 bit_vals = [parse(Int64,string(b)) for b in bs]
 instance = reshape(BitArray(bit_vals), 1, num_vars)
 # println(instance)
 # println(size(instance))
 push!(instances, instance)
end
println("Examples Generated...")
uq = vcat(instances...)
println("Size of instances : $(size(uq))")
uq = DataFrame((BitArray(Base.convert(Matrix{Bool}, uq))))
println("Size of instances : $(size(uq))")

train_ll = ll(uq)

for pc in pcs
 println(pc)
 println((exp(logsumexp(log_likelihood_per_instance(pc, uq)))))
 println("---")
end

lse = logsumexp(train_ll)
se = exp(lse)

println(parsed_args["logdir"])
println(parsed_args["dataset"])
println(parsed_args["run_name"])
println("Logsumexp : $lse")
println("Sumexmp : $(se)")

open(joinpath("bin", "sum_probs_exhaust.txt"), "a") do file
 write(file, parsed_args["logdir"])
 write(file, "\n")
 write(file, parsed_args["dataset"])
 write(file, "\n")
 write(file, parsed_args["run_name"])
 write(file, "\n")
 write(file, string("Logsumexp : ", string(lse)))
 write(file, "\n")
 write(file, string("Sumexp : ", string(se)))
 write(file, "\n\n\n")
end
# end





# if parsed_args["exhaust"]
#  exhaust(parsed_args)
# else
#  non_exhaust(parsed_args)
# end
