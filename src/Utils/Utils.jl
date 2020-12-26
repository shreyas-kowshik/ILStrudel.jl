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
using Statistics

export

# Heuristics
split_heuristic,
clone_heuristic,

# IndependenceTest
_mutual_information # '_' since it conflicts with `ProbabilisticCircuits.jl`'s definition

include("Heuristics.jl")
include("IndepdenceTest.jl")

end