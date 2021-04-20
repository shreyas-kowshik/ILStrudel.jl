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

    save_as_csv(summary_dict; filename=joinpath(log_path, "summary.csv"), header=header)
end

function jld_summary(log_path)
    summary_dict = Dict()
    dsets = readdir(log_path)
    init_keys = false

    function chk_type(d, k)
    	if isa(d[k], Number) || isa(d[k], String)
		    return true
	    else
		    return false
	    end
    end
    
    for dset in dsets
    	if !(dset in DSETS)
		continue
	end

        dset_path = joinpath(log_path, dset)

        for log in readdir(dset_path)
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

            for k in keys(d)
	    	    if !chk_type(d, k)
			        continue
		        end

                if isnothing(d[k])
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

                    write(file, joinpath(dset_path, log))
		            write(file, "\n\n\n")
                    d = load(joinpath(dset_path, log))["config_dict"]
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
                    write(file, "Sum_x p_z_given_x .* bitmasks : ")
		            write(file, "\n\n\n")
                    write(file, string(sum(em_data_weights .* b, dims=1)))
		            write(file, "\n\n\n")
                    write(file, "-------------------------------")
		            write(file, "\n\n\n")
                end
            end
        end
    end
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
