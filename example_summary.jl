using Pkg
Pkg.activate("/media/shreyas/Data/UCLA-Intern/ILStrudel/ILStrudel.jl")

using Statistics
using ArgParse
using JLD
using CSV
using DataFrames

DSETS = ["nltcs", "msnbc", "kdd", "plants", "baudio", "jester", "bnetflix", "accidents", 
"tretail", "pumsb_star", "dna", "kosarek", "msweb", "book", "tmovie",
"cwebkb", "cr52", "c20ng", "bbc", "ad"]

function save_as_csv(dict; filename, header=keys(dict))
    table = DataFrame(;[Symbol(x) => dict[x] for x in header]...)
    CSV.write(filename, table; )
end

function create_summary(log_path, header)
    summary_dict = Dict()
    dsets = readdir(log_path)
    test_lls = []
    sizes = []

    for dset in DSETS
        if dset in dsets
            dset_path = joinpath(log_path, dset)
            best_test_ll = -Inf
            best_size = 1
            
            for log in readdir(dset_path)
                if !occursin(".jld", log)
                    continue
                end

                if occursin("bitmask", log)
                    continue
                end

                println(joinpath(dset_path, log))
                d = load(joinpath(dset_path, log))["config_dict"]
                test_ll = d["test_ll"]
                if test_ll > best_test_ll
                    best_test_ll = test_ll
                    best_size = d["params"]
                end
            end
            
            push!(test_lls, best_test_ll)
            push!(sizes, best_size)
        else
            push!(test_lls, Inf)
            push!(sizes, -1)
        end
    end

    summary_dict["dataset"] = DSETS
    summary_dict["test_ll"] = test_lls
    summary_dict["size"] = sizes

    println("End of create_summary")

    save_as_csv(summary_dict; filename=joinpath(log_path, "summary.csv"), header=header)
    println("Saved CSV file")

end

function jld_summary(log_path)
    summary_dict = Dict()
    dsets = readdir(log_path)
    init_keys = false

    function chk_type(d, k)
    	if isa(d[k], Number) || isa(d[k], String) || k=="bitmask_path"
		    return true
	    else
		    return false
	    end
	    return true
    end
    
    for dset in dsets
    	if !(dset in DSETS)
		continue
	end

        dset_path = joinpath(log_path, dset)

        for log in sort(readdir(dset_path))
	    	if !occursin(".jld", log)
			    continue
		    end

            if occursin("bitmask", log)
                continue
            end

            println(joinpath(dset_path, log))
            d = load(joinpath(dset_path, log))["config_dict"]

            if !init_keys
                for k in keys(d)
		    if chk_type(d, k)
	                    summary_dict[k] = []
		    end
                end
                init_keys = true
            end

            for k in keys(summary_dict)
	    	    if !(chk_type(d, k))
			        continue
		        end
                if isnothing(d[k])
		    println("Pushing NULL")
                    push!(summary_dict[k], "NULL")
                else
                    push!(summary_dict[k], d[k])
                end
            end
        end
    end

    save_as_csv(summary_dict; filename=joinpath(log_path, "runs.csv"))
end

"""
Prints some statistics for analysis in a `.txt` file for the run
"""
function print_stats(log_path)
    summary_dict = Dict()
    dsets = readdir(log_path)
    test_lls = []
    sizes = []

    open(joinpath(log_path, "analysis.txt"), "w") do file
        for dset in DSETS
            if dset in dsets
                dset_path = joinpath(log_path, dset)
                for log in readdir(dset_path)
                    if !occursin(".jld", log)
                        continue
                    end

                    if occursin("bitmask", log)
                        continue
                    end

		    # Load pmi stats

                    write(file, joinpath(dset_path, log))
		            write(file, "\n\n\n")
                    d = load(joinpath(dset_path, log))["config_dict"]
		    
		    mi_fname = string("mi", d["num_mi_bags"], "s")
		    pmi_stats = load(joinpath("bin", "pmi_stats", dset, string(mi_fname, ".jld")))[mi_fname]
		    mu = mean(pmi_stats)
		    sigma = std(pmi_stats)
		    
                    b = load(joinpath(dset_path, "bitmasks.jld"))["bitmasks"]
                    b = hcat(b...)
                    em_data_weights = d["em_data_weights"]
                    

                    write(file, "Sum_x p_z_given_x : ")
		            write(file, "\n\n\n")
                    write(file, string(sum(em_data_weights, dims=1)))
		            write(file, "\n\n\n")
                    write(file, "Sum_x bitmasks : ")
		            write(file, "\n\n\n")
                    write(file, string(sum(b, dims=1)))
		            write(file, "\n\n\n")
		    
		    # Descriptive Statistics
                    write(file, "Sum_x p_z_given_x .* bitmasks : ")
		            write(file, "\n\n\n")
                    write(file, string(sum(em_data_weights .* b, dims=1)))
		            write(file, "\n\n\n")
                    write(file, "Mean: ")
		            write(file, "\n")
                    write(file, string(mean(em_data_weights .* b, dims=1)))
		            write(file, "\n")
                    write(file, "Std: ")
		            write(file, "\n")
                    write(file, string(std(em_data_weights .* b, dims=1)))
		            write(file, "\n")
                    write(file, "Minimum: ")
		            write(file, "\n")
                    write(file, string(minimum(em_data_weights .* b, dims=1)))
		            write(file, "\n")
                    write(file, "Maximum: ")
		            write(file, "\n")
                    write(file, string(maximum(em_data_weights .* b, dims=1)))
		            write(file, "\n")
		    # quantile!(freq_arr, 0.25)
		    emb = em_data_weights .* b
		    q_arr = [emb[:, i] for i in 1:size(emb)[2]]
		    q25 = quantile!.(q_arr, 0.25)
		    q50 = quantile!.(q_arr, 0.50)
		    q75 = quantile!.(q_arr, 0.75)

                    write(file, "Quantile 0.25: ")
		            write(file, "\n")
                    write(file, string(q25))
		            write(file, "\n")
                    write(file, "Quantile 0.50: ")
		            write(file, "\n")
                    write(file, string(q50))
		            write(file, "\n")
                    write(file, "Quantile 0.75: ")
		            write(file, "\n")
                    write(file, string(q75))
		            write(file, "\n\n\n")




		    write(file, "p_z_given_x summary statistics\n\n\n");
                    write(file, string(sum(em_data_weights, dims=1)))
		            write(file, "\n\n\n")
                    write(file, "Mean: ")
		            write(file, "\n")
                    write(file, string(mean(em_data_weights, dims=1)))
		            write(file, "\n")
                    write(file, "Std: ")
		            write(file, "\n")
                    write(file, string(std(em_data_weights, dims=1)))
		            write(file, "\n")
                    write(file, "Minimum: ")
		            write(file, "\n")
                    write(file, string(minimum(em_data_weights, dims=1)))
		            write(file, "\n")
                    write(file, "Maximum: ")
		            write(file, "\n")
                    write(file, string(maximum(em_data_weights, dims=1)))
		            write(file, "\n")
		    # quantile!(freq_arr, 0.25)
		    emb = em_data_weights
		    q_arr = [emb[:, i] for i in 1:size(emb)[2]]
		    q25 = quantile!.(q_arr, 0.25)
		    q50 = quantile!.(q_arr, 0.50)
		    q75 = quantile!.(q_arr, 0.75)

                    write(file, "Quantile 0.25: ")
		            write(file, "\n")
                    write(file, string(q25))
		            write(file, "\n")
                    write(file, "Quantile 0.50: ")
		            write(file, "\n")
                    write(file, string(q50))
		            write(file, "\n")
                    write(file, "Quantile 0.75: ")
		            write(file, "\n")
                    write(file, string(q75))
		            write(file, "\n\n\n")



		    write(file, "pmi threshold\n\n\n")
		    write(file, string(d["pmi_thresh"]))
		    write(file, "\n\n\n")
		    write(file, "pmis\n\n\n")
		    write(file, string(d["pmis"]))
		    write(file, "\n\n\n")
		    
		    write(file, "num_mi_bags : ")
		    write(file, string(d["num_mi_bags"]))
		    write(file, "\n")
		    write(file, "mu : ")
		    write(file, string(mu))
		    write(file, "\n")
		    write(file, "sigma : ")
		    write(file, string(sigma))
		    write(file, "\n")
		    
		    write(file, "\n\n\n")
		    normalized_pmis = (d["pmis"] .- mu) ./ sigma
		    write(file, string(normalized_pmis))
		    write(file, "\n\n\n")
		    
                    write(file, "-------------------------------")
		            write(file, "\n\n\n")
                end
            end
        end
    end
end

"""
Usage : `julia1 example_summary.jl --logdir /space/shreyas-kowshik/runs/bag2`
"""
function combine_seeds(path)
 prefix = string(path, "_")
 files = readdir("/space/shreyas-kowshik/runs")
 summary_dict = Dict()
 vals = zeros(20)
 counts = 0
 header = []
 push!(header, "dataset")
 push!(header, "average")
 
 for file in files
  println(file)
  if !occursin(prefix, joinpath("/space/shreyas-kowshik/runs", file))
   continue
  end

  summary_path = joinpath("/space/shreyas-kowshik/runs", file, "summary.csv")
  println(summary_path)
  d = CSV.read(summary_path, DataFrame)

  summary_dict["dataset"] = d["dataset"]
  summary_dict[string("seed_", file[end-2:end])] = d["test_ll"]
  push!(header, string("seed_", file[end-2:end]))
  vals .= vals .+ d["test_ll"]
  counts += 1
 end

 vals = vals ./ counts
 summary_dict["average"] = vals

 if !isdir(path)
  mkpath(path)
 end

 filename = joinpath(path, "summary.csv")
 save_as_csv(summary_dict; filename, header=header)
end


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--logdir"
            help = "Name of the dataset"
            arg_type = String
            required = false
            default = joinpath("/space/shreyas-kowshik/ILStrudel/log")
    end

    return parse_args(s)
end

parsed_args = parse_commandline()
header = ["dataset", "test_ll", "size"]
create_summary(parsed_args["logdir"], header)

jld_summary(parsed_args["logdir"])
print_stats(parsed_args["logdir"])

#combine_seeds(parsed_args["logdir"])
