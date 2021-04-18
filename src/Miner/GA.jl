using Distributions
using Evolutionary

"""
Contains functions for mining Context-Specific-Independences at the root level using Genetic Algorithms
"""

### Fitness Function ###
function fitness(dmat, uq, dict, prime_lits=[1], sub_lits=[2]; idx=BitArray(ones(1)), thresh = 0.1)
    function score(bitmask::BitArray)
        bm = BitArray(zeros(size(uq)[1]))
        bm[idx] .= bitmask

        bm_mi = BitArray(zeros(size(dmat)[1]))
        instances = uq[bm, :]
        for i in 1:size(instances)[1]
            bm_mi[dict[instances[i, :]]] .= 1
        end

        mi = bootstrap_mutual_information(dmat[bm_mi, :], prime_lits, sub_lits; use_gpu=true, k=1, α=1.0)
   	# println("Verbose : $mi") 
        if mi < thresh && sum(bm) > 0
            return -1.0 * sum(bm)
        end
        return 1.0
    end
    return score
end

### Crossover ###
function rand_from_start(bm1::BitArray, bm2::BitArray)
    r = rand(1:size(bm1)[1])
    tem1 = copy(bm1[1:r])
    tem2 = copy(bm2[1:r])
    bm2[1:r] .= tem1
    bm1[1:r] .= tem2
    return bm1, bm2
end

function bm_crossover(; type::String="rand_from_start")
    if type == "rand_from_start"
        return rand_from_start
    else
        error("Type of crossover not defined")
    end
end

### Mutation ###
function rand_idx(p::Real = 0.3)
    function mutation(bm::BitArray)
        n = size(bm)[1]
        d = Binomial(1, p)
        mutation_indices = BitArray(rand(d, n))
        bm[mutation_indices] .= .!(bm[mutation_indices])
        return bm
    end
    return mutation
end

function bm_mutation(; type::String="rand_idx", p=0.3)
    if type == "rand_idx"
        return rand_idx(p)
    else
        error("Type of mutation not defined")
    end
end

### Initial Population ###
import Evolutionary.initial_population
function initial_population(method::M, individual::BitArray) where {M<:Evolutionary.AbstractOptimizer}
    population = []
    s = Int(floor(size(individual)[1] / Evolutionary.population_size(method)))
    for i in 1:Evolutionary.population_size(method)
    	n = size(individual)
        d = Binomial(1, 0.5)
        bm = BitArray(rand(d, n))
	# This is really sparse
        # bm = BitArray(zeros(size(individual)))
        # bm[(i - 1) * s + 1] = 1
        push!(population, bm)
    end
    return population
end

"""
Mine CSIs at the root level
"""
function mine_csi_root_ga(pc, vtree, train_x, num_samples;
                    iterations=10, mutation_prob=0.1, population_size=100,
                    pmi_thresh=0.1)

    # N = size(train_x)[1]
    dmat = BitArray(convert(Matrix, train_x))

    and = children(pc)[1]
    og_lits = collect(Set{Lit}(variables(and.vtree))) # All literals

    prime_lits = sort([abs(l) for l in og_lits if l in variables(children(and)[1].vtree)])
    sub_lits = sort([abs(l) for l in og_lits if l in variables(children(and)[2].vtree)])

    prime_lits = sort(collect(Set{Lit}(prime_lits)))
    sub_lits = sort(collect(Set{Lit}(sub_lits)))
    prime_sub_lits = sort([prime_lits..., sub_lits...])

    # Unique mapping
    uq = unique(train_x)
    dict = Dict()
    for i in 1:size(uq)[1]
        dict[uq[i, :]] = []
    end

    for i in 1:size(train_x)[1]
        push!(dict[train_x[i, :]], i)
    end

    N = size(uq)[1]

    # Start Optimization #
    opts = Evolutionary.Options(iterations=iterations, show_every=1, show_trace=true, store_trace=true,
                           successive_f_tol=100000)
    
    algo = GA(
    selection = rouletteinv,
    mutation =  bm_mutation(type="rand_idx", p=mutation_prob),
    crossover = bm_crossover(),
    mutationRate = 0.95,
    crossoverRate = 0.95,
    populationSize = population_size,
    ε = 0.2
    )

    seeds = []
    for i in 1:num_samples
        push!(seeds, i * 100 + 47)
    end

    bitmasks = []
    acc = BitArray(zeros(N))

    for seed in seeds
        Random.seed!(seed);
        idx = .!(acc)
        println(sum(idx))

        if sum(idx) == 0
            break
        end
        
        bm = BitArray(zeros(sum(idx)))
        bm[1] = 1
        
        res = Evolutionary.optimize(fitness(dmat, uq, dict, prime_lits, sub_lits; idx=idx, thresh=pmi_thresh),
                                    bm, algo, opts)
        evomodel = Evolutionary.minimizer(res)
        bitmask = BitArray(zeros(N))
        bitmask[idx] = evomodel
        
        acc = acc .| bitmask
        
        # push!(bitmasks, bitmask)
        bm_train_x = BitArray(zeros(size(dmat)[1]))
        instances = uq[bitmask, :]
        for i in 1:size(instances)[1]
            bm_train_x[dict[instances[i, :]]] .= 1
        end
        push!(bitmasks, bm_train_x)
    end

    # Assign remaining values to a bitmask
    # push!(bitmasks, .!(acc))

    if sum(.!(acc)) > 0
        bm_train_x = BitArray(zeros(size(dmat)[1]))
        instances = uq[.!(acc), :]
        for i in 1:size(instances)[1]
            bm_train_x[dict[instances[i, :]]] .= 1
        end
        push!(bitmasks, bm_train_x)
    end

    return bitmasks
end
