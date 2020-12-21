"""
Module for Data Mining.
    - Genetic Algorithm (GA) CSI miner
"""

module Miner

# Do these imports if you use functions of these modules inside this module
using ..Utils

# Do these imports if you use functions of these packages inside this module
using LogicCircuits
using ProbabilisticCircuits
using Random
using Statistics

export
mine_csi_root_ga

include("GA.jl")

end