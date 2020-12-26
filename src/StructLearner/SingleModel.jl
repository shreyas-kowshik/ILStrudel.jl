
function clone_step(circuit::Node; loss=nothing, depth=1)
    (or, and1, and2) = loss(circuit)

    if or == nothing || and1 == nothing || and2 == nothing
        return circuit
    end
    
    clone(circuit, and1, and2, or; depth=depth)
end

# Custom Struct Learn export
function _struct_learn(circuit::Node; 
    primitives=[split_step], 
    kwargs=Dict(split_step=>(loss=random_split, depth=0)),
    maxiter=typemax(Int), stop::Function=x->false)

    for iter in 1 : maxiter
        # primitive_step = rand(primitives)

        if iter % 2 == 0
            primitive_step = primitives[2]
        else
            primitive_step = primitives[1]
        end

        # kwarg = kwargs[primitive_step]
        c2, _ = primitive_step(circuit)

        if stop(c2)
            return c2
        end
        circuit = c2
    end
    circuit
end



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
    split_loss(circuit) = split_heuristic(circuit, train_x; pick_edge=pick_edge, pick_var=pick_var)
    clone_loss(circuit) = clone_heuristic(circuit, train_x)

    pc_split_step(circuit) = begin
        c::ProbCircuit, = split_step(circuit; loss=split_loss, depth=depth, sanity_check=sanity_check)
        if batch_size > 0
            estimate_parameters(c, batch(train_x, batch_size); pseudocount, use_gpu)
        else
            estimate_parameters(c, train_x; pseudocount, use_gpu)
        end
        return c, missing
    end

    pc_clone_step(circuit) = begin
        c::ProbCircuit = clone_step(circuit; loss=clone_loss, depth=depth)
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
        # push!(valid_lls_track, valid_ll)
        # if length(valid_lls_track) > 30
        #     if valid_lls_track[end] - valid_lls_track[end - 30] < 0
        #         println("Early Stopping")
        #         return true
        #     end
        # end
        false
    end

    log_per_iter(pc)
    pc = _struct_learn(pc;
        primitives=[pc_split_step, pc_clone_step], kwargs=Dict(pc_split_step=>(), pc_clone_step=>()),
        maxiter=maxiter, stop=log_per_iter)

    if return_vtree
        pc, vtree
    else
        pc
    end
end
