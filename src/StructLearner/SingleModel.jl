
function learn_single_model(dataset_name::String)
    train_x, valid_x, test_x = twenty_datasets(dataset_name)
    return learn_single_model(train_x, valid_x, test_x)
end

function learn_single_model(train_x, valid_x, test_x; 
    pick_edge="eFlow", pick_var="vMI", depth=1, 
    pseudocount=1.0,
    sanity_check=true,
    maxiter=100,
    seed=nothing,
    return_vtree=false)

    # Initial Structure
    pc, vtree = learn_chow_liu_tree_circuit(train_x)

    learn_single_model(train_x, valid_x, test_x,
                pc, vtree; pick_edge, pick_var, depth, pseudocount, sanity_check,
                maxiter, seed, return_vtree)
end

function learn_single_model(train_x, valid_x, test_x, pc, vtree;
    pick_edge="eFlow", pick_var="vMI", depth=1,
    pseudocount=1.0,
    sanity_check=true,
    maxiter=100,
    seed=nothing,
    return_vtree=false,
    batch_size=0,
    use_gpu=false)

    if seed !== nothing
        Random.seed!(seed)
    end

    # structure_update
    loss(circuit) = split_heuristic(circuit, train_x; pick_edge=pick_edge, pick_var=pick_var)

    pc_split_step(circuit) = begin
        c::ProbCircuit, = split_step(circuit; loss=loss, depth=depth, sanity_check=sanity_check)
        if batch_size > 0
            estimate_parameters(c, batch(train_x, batch_size); pseudocount, use_gpu)
        else
            estimate_parameters(c, train_x; pseudocount, use_gpu)
        end
        return c, missing
    end

    iter = 0
    count = 0
    valid_lls_track = []
    log_per_iter(circuit) = begin
        # ll = EVI(circuit, train_x);
        if batch_size > 0
            train_ll = log_likelihood_avg(circuit, batch(train_x, batch_size); use_gpu)
            valid_ll = log_likelihood_avg(circuit, batch(valid_x, batch_size); use_gpu)
            test_ll = log_likelihood_avg(circuit, batch(test_x, batch_size); use_gpu)
        else
            train_ll = log_likelihood_avg(circuit, train_x; use_gpu)
            valid_ll = log_likelihood_avg(circuit, valid_x; use_gpu)
            test_ll = log_likelihood_avg(circuit, test_x; use_gpu)
        end
        println("Iteration $iter/$maxiter. nodes = $(num_nodes(circuit)); params = $(num_parameters(circuit))")
        println("TrainLL = $(train_ll); ValidLL = $(valid_ll); TestLL = $(test_ll)");
        iter += 1

        # Early Stopping
        push!(valid_lls_track, valid_ll)
        # if length(valid_lls_track) > 30
        #     if valid_lls_track[end] - valid_lls_track[end - 30] < 0
        #         println("Early Stopping")
        #         return true
        #     end
        # end
        false
    end

    log_per_iter(pc)
    pc = struct_learn(pc;
        primitives=[pc_split_step], kwargs=Dict(pc_split_step=>()),
        maxiter=maxiter, stop=log_per_iter)

    if return_vtree
        pc, vtree
    else
        pc
    end
end
