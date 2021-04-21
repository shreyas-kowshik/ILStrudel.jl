
function learn_mine_ensemble(train_x, valid_x, test_x;
    mine_iterations=3,
    population_size=100,
    num_mine_samples=10,
    pick_edge="eFlow", pick_var="vMI", depth=1, 
    pseudocount=1.0,
    sanity_check=true,
    maxiter=100,
    seed=nothing,
    return_vtree=false,
    return_bitmasks=false,
    pmi_thresh=0.1,
    bitmasks=nothing)

    pc, vtree = learn_chow_liu_tree_circuit(train_x)

    if isnothing(bitmasks)
        bitmasks = mine_csi_root_ga(pc, vtree, train_x, num_mine_samples; 
                                    iterations=mine_iterations, population_size=population_size,
                                    pmi_thresh=pmi_thresh)
    end

    # For pMI computation #
    dmat = BitArray(convert(Matrix, train_x))

    and = children(pc)[1]
    og_lits = collect(Set{Lit}(variables(and.vtree))) # All literals

    prime_lits = sort([abs(l) for l in og_lits if l in variables(children(and)[1].vtree)])
    sub_lits = sort([abs(l) for l in og_lits if l in variables(children(and)[2].vtree)])

    prime_lits = sort(collect(Set{Lit}(prime_lits)))
    sub_lits = sort(collect(Set{Lit}(sub_lits)))
    prime_sub_lits = sort([prime_lits..., sub_lits...])
    ##################

    circuits = []
    final_bitmasks = []
    final_pmis = []
    for bitmask in bitmasks
        println("Size of Bitmask : $(sum(bitmask))")
        println("$(size(bitmask))")

        if(sum(bitmask) == 0)
            continue
        end
        push!(final_bitmasks, bitmask)

        # Comput pMI
        println("Threshold pMI : $pmi_thresh")
        pMI = bootstrap_mutual_information(dmat, prime_lits, sub_lits; use_gpu=true, k=1, α=1.0) 
        println("Bitmask pMI : $pMI")
        push!(final_pmis, pMI)

        bitmask = BitArray(bitmask)
        pc = learn_single_model(train_x[bitmask, :], valid_x, test_x; 
        pick_edge=pick_edge, pick_var=pick_var, depth=depth, 
        pseudocount=pseudocount,
        sanity_check=true,
        maxiter=maxiter,
        seed=nothing,
        return_vtree=false)

        push!(circuits, pc)
    end
    bitmasks = copy(final_bitmasks)

    if return_bitmasks
        return circuits, bitmasks, final_pmis
    else
        circuits
    end
end
