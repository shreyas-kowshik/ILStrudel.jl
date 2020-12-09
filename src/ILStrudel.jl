module ILStrudel

using Reexport # For using external modules

# Include child modules
include("Utils/Utils.jl")

# Use Child Modules (To reexport functions)
@reexport using .Utils



end # module
