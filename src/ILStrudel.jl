module ILStrudel

using Reexport # For using external modules

# Include child modules
include("Utils/Utils.jl")
include("StructLearner/StructLearner.jl")
include("Miner/Miner.jl")

# Use Child Modules (To reexport functions)
@reexport using .Utils
@reexport using .StructLearner
@reexport using .Miner


end # module
