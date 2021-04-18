using Combinatorics
using CUDA
using LinearAlgebra


############################### Mutual Information ###############################
"""
k-way pMI computation
Works only for k={1,2} for now
"""
function kway_MI_cpu(dmat, vars_x, vars_y; k=2, α=1.0)
    num_prime_vars = length(vars_x)
    num_sub_vars = length(vars_y)
    num_vars = num_prime_vars + num_sub_vars

    N = size(dmat)[1]

    """
    Loop over NCk on primes
    Loop over NCk on subs
    Compute pMI for the current subsets selected
    """
    k1 = minimum([k, length(vars_x)])
    k2 = minimum([k, length(vars_y)])

    cnt = 0
    pMI_val = 0.0
    for primes in combinations(vars_x, k1)
        for subs in combinations(vars_y, k2)
            # Assume k=2 for now #
            # Generalise later if required #

            # x1x2 : Prime Variables
            # y1y2 : Sub Variables
            prime_mat_vals = []
            sub_mat_vals = []
            vec_x1x2 = mapreduce(x->x, &, dmat[:, Var.(primes)], dims=[2])
            push!(prime_mat_vals, vec_x1x2)

            if k1 > 1
                d = dmat[:, Var.(primes)]
                d[:, 1] = .!(d[:, 1])
                vec_nx1x2 = mapreduce(x->x, &, d, dims=[2])
                push!(prime_mat_vals, vec_nx1x2)

                d = dmat[:, Var.(primes)]
                d[:, 2] = .!(d[:, 2])
                vec_x1nx2 = mapreduce(x->x, &, d, dims=[2])
                push!(sub_mat_vals, vec_x1nx2)
            end

            d = dmat[:, Var.(primes)]
            d = .!(d)
            vec_nx1nx2 = mapreduce(x->x, &, d, dims=[2])
            push!(prime_mat_vals, vec_nx1nx2)

            vec_y1y2 = mapreduce(x->x, &, dmat[:, Var.(subs)], dims=[2])
            push!(sub_mat_vals, vec_y1y2)

            if k2 > 1
                d = dmat[:, Var.(subs)]
                d[:, 1] = .!(d[:, 1])
                vec_ny1y2 = mapreduce(x->x, &, d, dims=[2])
                push!(sub_mat_vals, vec_ny1y2)

                d = dmat[:, Var.(subs)]
                d[:, 2] = .!(d[:, 2])
                vec_y1ny2 = mapreduce(x->x, &, d, dims=[2])
                push!(sub_mat_vals, vec_y1ny2)
            end

            d = dmat[:, Var.(subs)]
            d = .!(d)
            vec_ny1ny2 = mapreduce(x->x, &, d, dims=[2])
            push!(sub_mat_vals, vec_ny1ny2)

            for pval in prime_mat_vals
                for sval in sub_mat_vals
                    pcomb = sum(pval .& sval)/N
                    pprimes = sum(pval)/N
                    psubs = sum(sval)/N

                    pMI_val += (pcomb * (log((pcomb / ((pprimes * psubs) + 1e-6)) + 1e-6)))
		    cnt += 1
                end
            end

            ######################
        end
    end

    return pMI_val / (cnt + 1e-6)
end

function pMI_kernel_gpu(marginals, p_s, notp_s, p_nots, notp_nots,
    pMI_vec, num_prime_vars, num_sub_vars)
    index_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index_y = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (index_x > num_prime_vars) || (index_y > num_sub_vars)
        return nothing
    end

    pMI_vec[index_x, index_y] = 0.0

    if index_x == index_y
        return nothing
    end

    pMI_vec[index_x, index_y] += (p_s[index_x, index_y] * CUDA.log((p_s[index_x, index_y])/(marginals[index_x] * marginals[index_y] + 1e-13) + 1e-13))
    pMI_vec[index_x, index_y] += (notp_s[index_x, index_y] * CUDA.log((notp_s[index_x, index_y])/((1.0 - marginals[index_x]) * marginals[index_y] + 1e-13) + 1e-13))
    pMI_vec[index_x, index_y] += (p_nots[index_x, index_y] * CUDA.log((p_nots[index_x, index_y])/(marginals[index_x] * (1.0 - marginals[index_y]) + 1e-13) + 1e-13))
    pMI_vec[index_x, index_y] += (notp_nots[index_x, index_y] * CUDA.log((notp_nots[index_x, index_y])/((1.0 - marginals[index_x]) * (1.0 - marginals[index_y]) + 1e-13) + 1e-13))

    return nothing
end

function pMI_gpu(dmat, vars_x, vars_y; α=1.0)
    num_prime_vars = length(vars_x)
    num_sub_vars = length(vars_y)
    num_vars = num_prime_vars + num_sub_vars

    pMI_vec = to_gpu(zeros(num_vars, num_vars))

    num_threads = (16, 16)
    num_blocks = (ceil(Int, num_vars/16), ceil(Int, num_vars/16))
    N = size(dmat)[1]

    dummy = ones(num_prime_vars+num_sub_vars,num_prime_vars+num_sub_vars)
    d_d = cu(similar(dummy))
    d_nd = cu(similar(dummy))
    nd_nd = cu(similar(dummy))
    dmat_gpu = cu(dmat)
    dmat_tr_gpu = cu(collect(dmat'))
    not_dmat_gpu = cu(.!(dmat))
    not_dmat_tr_gpu = cu(collect((.!(dmat))'))

    mul!(d_d, dmat_tr_gpu, dmat_gpu)
    mul!(d_nd, dmat_tr_gpu, not_dmat_gpu)
    mul!(nd_nd, not_dmat_tr_gpu, not_dmat_gpu)

    d_d = (d_d .+ (α)) ./ (N + 4.0 * α)
    d_nd = (d_nd .+ (α)) ./ (N + 4.0 * α)
    nd_nd = (nd_nd .+ (α)) ./ (N + 4.0 * α)
    marginals = (dropdims(count(dmat, dims=1), dims=1) .+ (2.0 * α)) ./ (N + 4.0 * α)

    p_s = d_d
    p_nots = d_nd
    notp_s = collect(d_nd')
    notp_nots = nd_nd

    @cuda threads=num_threads blocks=num_blocks pMI_kernel_gpu(to_gpu(marginals),
                            p_s, to_gpu(notp_s), p_nots, notp_nots,
                            pMI_vec, num_vars, num_vars)

    cpu_pMI = to_cpu(pMI_vec)
    cpu_pMI = cpu_pMI[Var.(vars_x), Var.(vars_y)]
    cpu_pMI = mean(cpu_pMI)

    if abs(cpu_pMI) < 1e-10
        cpu_pMI = 0.0
    end

    return cpu_pMI
end

"""
Given bit-matrix `mat` compute empirical-mutual-information between two sets of variables `vars_x` and `vars_y`

k : Specifies number of variables from each set to approximate MI at a time
k=1 : pMI

k={1,2} only available for now
"""
function _mutual_information(mat, vars_x, vars_y; k=1, use_gpu=false, α=1.0)
    mi = 0.0
    vars = sort([vars_x..., vars_y...])
    var_map = Dict([k=>i for (i, k) in enumerate(vars)])
    vars_x = [var_map[v] for v in vars_x]
    vars_y = [var_map[v] for v in vars_y]
    mat = mat[:, vars]

    if k == 1
        if use_gpu == true
            mi = pMI_gpu(mat, vars_x, vars_y; α=α)
        else
            (_, pairwise_mi_mat) = mutual_information(mat; α=α)
            mi = mean(pairwise_mi_mat[vars_x, vars_y])
        end

    elseif k == 2
        if use_gpu == true
            error("GPU kernel not defined for k=$k")
        end

        mi = kway_MI_cpu(mat, vars_x, vars_y; α=α)
    else
        error("mutual_information not defined for k=$k")
    end

    # println("MI : $mi")
    return mi
end

function bootstrap_mutual_information(mat, vars_x, vars_y; num_bags=20, k=1, use_gpu=false, α=1.0)
    # println("MI Size : $(size(mat))")
    num_examples = size(mat)[1]

    # Value should ideally be zero but still passing it to code which should handle it
    if num_examples == 0
        return _mutual_information(mat, vars_x, vars_y; k=k, use_gpu=use_gpu, α=α)
    end

    MIs = []
    for i in 1:num_bags
		ids = rand(1:num_examples, num_examples)
        mat_bootstrap = copy(mat[ids, :])
        push!(MIs, _mutual_information(mat_bootstrap, vars_x, vars_y; k=k, use_gpu=use_gpu, α=α))
    end

    return mean(MIs)
end
