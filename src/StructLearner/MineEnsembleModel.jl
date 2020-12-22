
function learn_mine_ensemble(train_x, valid_x, test_x;
    mine_iterations=3,
    num_mine_samples=10,
    pick_edge="eFlow", pick_var="vMI", depth=1, 
    pseudocount=1.0,
    sanity_check=true,
    maxiter=100,
    seed=nothing,
    return_vtree=false)

    pc, vtree = learn_chow_liu_tree_circuit(train_x)
    bitmasks = mine_csi_root_ga(pc, vtree, train_x, num_mine_samples; iterations=mine_iterations)

    circuits = []
    for bitmask in bitmasks
        println("Size of Bitmask : $(sum(bitmask))")
        pc = learn_single_model(train_x[bitmask, :], valid_x[bitmask, :], test_x[bitmask, :]; 
        pick_edge="w_ind", pick_var="w_ind", depth=1, 
        pseudocount=1.0,
        sanity_check=true,
        maxiter=200,
        seed=nothing,
        return_vtree=false)

        push!(circuits, pc)
    end

    circuits
end
