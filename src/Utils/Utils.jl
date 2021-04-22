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

# Miscellaneous
generate_pmi_bagging_stats, generate_pmi_runtime_stats, plot_instance_frequency,

# IndependenceTest
_mutual_information, bootstrap_mutual_information # '_' since it conflicts with `ProbabilisticCircuits.jl`'s definition

include("Heuristics.jl")
include("IndepdenceTest.jl")
include("Misc.jl")

end
