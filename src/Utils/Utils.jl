"""
Module for Utilities including :
    - Transformations (any additions not in current depenencies)
    - Plotting
    - Logging
    - Heuristics (for primitive operations)
    - Independence Testing
"""

module Utils

# Do these imports if you use functions of these packages inside this module
using LogicCircuits
using ProbabilisticCircuits
using Combinatorics

"""
export

.
"""
export

# Heuristics
split_heuristic

"""
include(...)
"""
include("Heuristics.jl")
include("IndepdenceTest.jl")

end