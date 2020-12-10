using Pkg
Pkg.activate("/media/shreyas/Data/UCLA-Intern/ILStrudel/ILStrudel.jl")

using Jive

runtests(@__DIR__, skip=["runtests.jl", "helper", "_manual_"])