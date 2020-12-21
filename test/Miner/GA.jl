using Pkg
Pkg.activate("/media/shreyas/Data/UCLA-Intern/ILStrudel/ILStrudel.jl")

using Test
using LogicCircuits
using ProbabilisticCircuits
using ILStrudel

@testset "Direct Call" begin
    train_x, valid_x, test_x = twenty_datasets("nltcs")
    dmat = BitArray(convert(Matrix, train_x))

    num_samples = 10
    pc, vtree = learn_chow_liu_tree_circuit(train_x)
    bitmasks = mine_csi_root_ga(pc, vtree, train_x, num_samples; iterations=3)

    for b1 in bitmasks
        for b2 in bitmasks
            if b1 == b2
                continue
            end
            @assert sum(b1 .& b2) == 0 "Not Disjoint"
        end
    end
end
