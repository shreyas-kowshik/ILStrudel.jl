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

likelihood_per_instance(m::Mixture, data; # weighted data not passed here, only passed during learning phase
                            pseudocount=1.0, batch_size=0, use_gpu=false) = begin
    N = size(data)[1]
    num_components = length(m.components)

    # Here Fs are storing the log-likelihoods
    F_t_1 = exp.(log_likelihood_per_instance(m.components[1], data))

    for i in 2:num_components
        α = m.weights[i]
        ht = exp.(log_likelihood_per_instance(m.components[i], data))
        Ft = ((1.0 - α) .* F_t_1) .+ (α .* ht)
        F_t_1 = copy(Ft)
    end

    total = sum(F_t_1)
    total2 = sum(log.(F_t_1))
    # @assert abs(1.0 - total) < 1e-6 "Total $total does not sum to 1.0"

    return F_t_1
end

function normalize(w)
    # w = w .- maximum(w)
    w = w ./ sum(w)
    w
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
        w = 1.0 ./ F_t_1
        w = normalize(w)
        weighted_train_x = add_sample_weights(train_x, w)

        pc, vtree = learn_weighted_chow_liu_tree_circuit(weighted_train_x)
        learn_single_model(weighted_train_x, valid_x, test_x, pc, vtree;
                           pseudocount=pseudocount,
                           maxiter=maxiter)

        likelihood = exp.(log_likelihood_per_instance(pc, train_x))

        # TODO check this condition later
        # sum_w_h = sum(exp.(log_likelihood_per_instance(pc, weighted_train_x)))
        
        # if sum_w_h <= N
        #     break
        # end

        α = 1.0
        min_score = Inf
        for step in 0.0:0.001:1.0
            score = -1.0 * (sum(log.((1.0 - step) .* F_t_1 .+ (step .* likelihood))))

            if score < min_score
                min_score = score
                α = step
            end
        end

        add_component(mixture, pc, α)
    end

    return mixture
end
