"""
Initialise weights
Learn single model on weighted data
Obtain coefficient
Add to ensemble
Update training weights
"""

mutable struct Mixture
    components # pcs
    weights # weights as normal, not log valued weights

    Mixture() = new([], [])
end

add_component(m::Mixture, pc, w) = begin
    push!(m.components, pc)
    push!(m.weights, w)
end

add_component(m::Mixture, pc) = begin
    push!(m.components, pc)
end

function normalize(w, N)
    # w = w .- maximum(w)
    # println("Normalize : Sum W : $(sum(w))")
    w = w ./ sum(w)
    w = w .* N
    w
end

# likelihood_per_instance(m::Mixture, data; # weighted data not passed here, only passed during learning phase
#                             pseudocount=1.0, batch_size=0, use_gpu=false) = begin
#     N = size(data)[1]
#     num_components = length(m.components)

#     # Here Fs are storing the log-likelihoods
#     F_t_1 = exp.(log_likelihood_per_instance(m.components[1], data))

#     for i in 2:num_components
#         α = m.weights[i]
#         ht = exp.(log_likelihood_per_instance(m.components[i], data))
#         Ft = ((1.0 - α) .* F_t_1) .+ (α .* ht)
#         F_t_1 = copy(Ft)
#     end

#     total = sum(F_t_1)
#     total2 = sum(log.(F_t_1))
#     # @assert abs(1.0 - total) < 1e-6 "Total $total does not sum to 1.0"

#     return F_t_1
# end

likelihood_per_instance(m::Mixture, data; # weighted data not passed here, only passed during learning phase
                            pseudocount=1.0, batch_size=0, use_gpu=false) = begin
        lls = hcat([log_likelihood_per_instance(pc, data) for pc in m.components]...)
        weights = reshape(m.weights, 1, length(m.weights))
        lls .= lls .+ log.(weights)
        sum(exp.(lls), dims=2)
end

function EM(m::Mixture, train_x; num_iters=5, pseudocount=1.0)
    # Initialise
    num_components = length(m.components)
    # component_weights = ones(num_components) ./ num_components
    component_weights = initial_weights(train_x, num_components) # Use library function
    component_weights = reshape(component_weights, 1, length(component_weights))

    # prev_val = Inf

    for iter in 1:num_iters
        println("EM Iter : $iter")

        # E Step
        log_p_x_given_z = hcat([log_likelihood_per_instance(pc, train_x) for pc in m.components]...)
        log_p_x_and_z = log_p_x_given_z .+ log.(component_weights)
        log_p_x = logsumexp(log_p_x_and_z, dims=2)

        ll_x = mean(log_p_x)
        # if prev_val - ll_x < 1e-3
        #     break
        # end
        # prev_val = ll_x
        println("Log_p_x : $(ll_x)")

        log_p_z_given_x = log_p_x_and_z .- log_p_x

        # M Step
        component_weights = sum(exp.(log_p_z_given_x), dims=1)
        component_weights = normalize(component_weights, 1.0)
        @assert abs(1.0 - sum(component_weights)) < 1e-6 "Parameters do not sum to 1 : $(sum(component_weights))"
        for i in 1:num_components
            weighted_train_x = add_sample_weights(copy(train_x), log_p_z_given_x[:, i])
            estimate_parameters(m.components[i], weighted_train_x; pseudocount=pseudocount)
        end
    end

    m.weights = copy(component_weights)
    return m
end

# DEBUG #
function print_ll_mixture(m::Mixture, train_x)
    println("-------------------")
    for pc in m.components
        ll = mean(log_likelihood_per_instance(pc, train_x))
        println(ll)
    end
    println("-------------------")
end
#########

# Trying to reproduce paper results
function boosting_shared_structure(train_x, valid_x, test_x, num_components;
    pseudocount=1.0, maxiter=100)
    N = size(train_x)[1]
    mixture = Mixture()
    pc, vtree = learn_chow_liu_tree_circuit(train_x)
    pc = learn_single_model(train_x, valid_x, test_x, pc, vtree;
                           pseudocount=pseudocount,
                           maxiter=maxiter)
    add_component(mixture, pc, 1.0)

    println("Boosting Started...")
    for i in 1:num_components
        pc2 = deepcopy(pc)
        println("Boosting Iteration : $i")
        F_t_1 = likelihood_per_instance(mixture, train_x)
        w = 1.0 ./ F_t_1
        w = normalize(w, N)
        weighted_train_x = add_sample_weights(copy(train_x), w)
        @assert isweighted(weighted_train_x) "Not weighted properly"

        estimate_parameters(pc2, weighted_train_x; pseudocount)
        likelihood = exp.(log_likelihood_per_instance(pc2, train_x))

        sum_w_h = sum(w .* log_likelihood_per_instance(pc2, train_x))
        println("Sum W H : $sum_w_h")

        α = 1.0
        min_score = Inf
        for step in 0.0:0.00001:1.0
            score = -1.0 * (mean(log.(((1.0 - step) .* F_t_1) .+ (step .* likelihood))))
            println("step : $step, Score : $score")

            if score < min_score
                min_score = score
                α = step
            end
        end

        println("α : $α")
        println("Current component log-likelihood : $(mean(log.(likelihood)))")
        println("Mixture log likelihood : $(mean(log.(F_t_1)))")
        print_ll_mixture(mixture, train_x)
        add_component(mixture, pc2, α)
    end

    return mixture
end

function boosting(train_x, valid_x, test_x, num_components;
                 pseudocount=1.0, maxiter=100)
    N = size(train_x)[1]
    mixture = Mixture()
    pc, vtree = learn_chow_liu_tree_circuit(train_x)
    add_component(mixture, pc, 1.0)

    println("Boosting Started")
    for i in 1:num_components
        println("Boosting Iteration : $i")
        F_t_1 = likelihood_per_instance(mixture, train_x)
        println("Mixture LL : $(mean(log.(F_t_1)))")
        # Sum w has to be N
        w = 1.0 ./ F_t_1
        # println("---W Stats---")
        # println("N : $N, Sum W : $(sum(w))")
        # println("Mean W : $(mean(w)), Variance W : $(var(w))")
        # w = Float32.(w .> thresh)
        # println("After threshold : $(sum(w))")
        w = normalize(w, N)
        println(size(w))
        println(size(train_x))
        weighted_train_x = add_sample_weights(copy(train_x), w)

        @assert isweighted(weighted_train_x) "Not weighted properly"

        pc, vtree = learn_weighted_chow_liu_tree_circuit(weighted_train_x)
        # pc, vtree = learn_chow_liu_tree_circuit(train_x)
        pc = learn_single_model(weighted_train_x, valid_x, test_x, pc, vtree;
                           pseudocount=pseudocount,
                           maxiter=maxiter)

        likelihood = exp.(log_likelihood_per_instance(pc, train_x))

        # TODO check this condition later
        sum_w_h = sum(exp.(log_likelihood_per_instance(pc, weighted_train_x)))
        
        println("Sum W H : $sum_w_h")
        # if sum_w_h <= N
        #     println("Sum W H : $sum_w_h")
        #     break
        # end

        # α = 1.0
        # min_score = Inf
        # for step in 0.0:0.00001:1.0
        #     score = -1.0 * (mean(log.(((1.0 - step) .* F_t_1) .+ (step .* likelihood))))

        #     if score < min_score
        #         min_score = score
        #         α = step
        #     end
        # end

        # println("α : $α")
        # add_component(mixture, pc, α)

        add_component(mixture, pc)
        mixture = EM(mixture, train_x; pseudocount=pseudocount)
    end

    return mixture
end
