module ILStrudel

using Reexport # For using external modules

# Include child modules
include("Utils/Utils.jl")
include("Miner/Miner.jl")
include("StructLearner/StructLearner.jl")

# Use Child Modules (To reexport functions)
@reexport using .Utils
@reexport using .Miner
@reexport using .StructLearner


end # module
