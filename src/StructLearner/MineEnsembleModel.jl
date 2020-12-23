
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
    return_bitmasks=false)

    pc, vtree = learn_chow_liu_tree_circuit(train_x)
    bitmasks = mine_csi_root_ga(pc, vtree, train_x, num_mine_samples; iterations=mine_iterations, population_size=population_size)

    circuits = []
    for bitmask in bitmasks
        println("Size of Bitmask : $(sum(bitmask))")
        println("$(size(bitmask))")
        bitmask = BitArray(bitmask)
        pc = learn_single_model(train_x[bitmask, :], valid_x, test_x; 
        pick_edge=pick_edge, pick_var=pick_var, depth=1, 
        pseudocount=pseudocount,
        sanity_check=true,
        maxiter=maxiter,
        seed=nothing,
        return_vtree=false)

        push!(circuits, pc)
    end

    if return_bitmasks
        return circuits, bitmasks
    else
        circuits
    end
end
