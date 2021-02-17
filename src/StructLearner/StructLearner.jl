"""
Module for Structure-Learning including :
    - Single-Model
    - Ensemble-Model
    - Mined-Ensemble-Model
"""

module StructLearner

# Do these imports if you use functions of these modules inside this module
using ..Utils
using ..Miner

# Do these imports if you use functions of these packages inside this module
using LogicCircuits
using ProbabilisticCircuits
using Random
using Statistics

export

# SingleModel
learn_single_model,
learn_weighted_chow_liu_tree_circuit,

# MineEnsembleModel
learn_mine_ensemble,

# BoostingModel
Mixture, add_component, likelihood_per_instance, boosting

include("SingleModel.jl")
include("MineEnsembleModel.jl")
include("BoostingModel.jl")

end