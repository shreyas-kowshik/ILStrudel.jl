using LinearAlgebra: diagind

"""
Pick the edge with maximum flow
"""
function eFlow(values, flows, candidates::Vector{Tuple{Node, Node}})
    edge2flows = map(candidates) do (or, and)
        count_downflow(values, flows, nothing, or, and)
    end
    (max_flow, max_edge_id) = findmax(edge2flows)
    candidates[max_edge_id], max_flow
end

"""
Pick the variable with maximum sum of mutual information
"""
function vMI(values, flows, edge, vars::Vector{Var}, train_x)
    examples_id = downflow_all(values, flows, num_examples(train_x), edge...)
    sub_matrix = train_x[examples_id, vars]
    (_, mi) = mutual_information(sub_matrix; α=1.0)
    mi[diagind(mi)] .= 0
    scores = dropdims(sum(mi, dims = 1), dims = 1)
    var = vars[argmax(scores)]
    score = maximum(scores)
    var, score
end

"""
Pick the edge randomly
"""
function eRand(candidates::Vector{Tuple{Node, Node}})
    return rand(candidates)
end

"""
Pick the variable randomly
"""
function vRand(vars::Vector{Var})
    return Var(rand(vars))
end

function w_ind(candidates::Vector{Tuple{Node, Node}}, values, flows, scope, train_x)
    # Convert to BitMatrix
    dmat = BitArray(convert(Matrix, train_x))
    N = num_examples(train_x)

    # Return parameters
    min_score = Inf
    edge = nothing
    var_ = nothing

    for (i, (or, and)) in enumerate(candidates)
        og_lits = collect(Set{Lit}(scope[and])) # All literals

        # On which you can split
        lits = sort(collect(intersect(filter(l -> l > 0, og_lits), - collect(filter(l -> l < 0, og_lits)))))
        vars = Var.(lits)

        prime_lits = sort([abs(l) for l in og_lits if l in scope[children(and)[1]]])
        sub_lits = sort([abs(l) for l in og_lits if l in scope[children(and)[2]]])

        prime_lits = sort(collect(Set{Lit}(prime_lits)))
        sub_lits = sort(collect(Set{Lit}(sub_lits)))
        prime_sub_lits = sort([prime_lits..., sub_lits...])

        @assert length(prime_lits) > 0 "Prime litset empty"
        @assert length(sub_lits) > 0 "Sub litset empty"
        prime_sub_vars = Var.(prime_sub_lits)
        lit_map = Dict(l => i for (i, l) in enumerate(prime_sub_lits))

        examples_id = downflow_all(values, flows, or, and)[1:N]

        if(sum(examples_id) == 0)
            continue
        end

        stotal = 0.0
        stotal = _mutual_information(dmat[examples_id, :], prime_lits, sub_lits)

        if stotal == 0.0
            continue
        end

        if(length(lits) == 0)
            continue
        end

        for j=1:length(lits)
            var = lits[j]
            pos_scope = examples_id .& dmat[:, var]
            neg_scope = examples_id .& (.!(pos_scope))
            @assert sum(examples_id) == (sum(pos_scope) + sum(neg_scope)) "Scopes do not add up"
            s1 = Inf
            s2 = Inf

            if sum(pos_scope) > 0
                stotal = _mutual_information(dmat[pos_scope, :], prime_lits, sub_lits)
            end
            if sum(neg_scope) > 0
                stotal = _mutual_information(dmat[pos_scope, :], prime_lits, sub_lits)
            end

            s = 0.0
            w1 = (sum(pos_scope)) / (1.0 * N)
            w2 = (sum(neg_scope)) / (1.0 * N)
            w = (sum(examples_id)) / (1.0 * N)

            if s1 == Inf
                s = (s2*w2) - (stotal*w)
            elseif s2 == Inf
                s = (s1*w1) - (stotal*w)
            else
                s = (s1*w1) + (s2*w2) - (2.0*stotal*w)
            end

            if s < min_score
                min_score = s
                edge = (or, and)
                var_ = var
            end
        end
    end

    return edge, var
end

"""
Params :
    - circuit, dataset

Returns :
    - (edge, var) to split on, logging_dict
"""
function split_heuristic(circuit::LogicCircuit, train_x; pick_edge="w_ind", pick_var="w_ind")
    # Boilerplate borrowed from LogicCircuits.jl
    if isweighted(train_x)
        train_x, weights = split_sample_weights(train_x)
    else
        weights = nothing
    end
    
    candidates, variable_scope = split_candidates(circuit)
    values, flows = satisfies_flows(circuit, train_x; weights = nothing) # Do not use samples weights here

    or = nothing
    and = nothing
    var = nothing

    if pick_edge == "w_ind" || pick_var == "w_ind"
        # Weighted heuristic
        (or, and), var = w_ind(candidates, values, flows, variable_scope, train_x)
    else
        if pick_edge == "eFlow"
            edge, flow = eFlow(values, flows, candidates)
        elseif pick_edge == "eRand"
            edge = eRand(candidates)
        else
            error("Heuristics $pick_edge to pick edge is undefined.")
        end

        or, and = edge
        vars = Var.(collect(variable_scope[and]))

        if pick_var == "vMI"
            var, score = vMI(values, flows, edge, vars, train_x)
        elseif pick_var == "vRand"
            var = vRand(vars)
        else
            error("Heuristics $pick_var to pick variable is undefined.")
        end
    end

    return (or, and), var
end