using JLD
t = [1, 2, 3]
a = []
push!(a, t)
push!(a, t)
BASE = homedir()
LOG_DIR = joinpath(BASE, "runs/mine_bagging_1/bnetflix")
if !isdir(LOG_DIR)
 mkpath(LOG_DIR)
end
LOG_DIR = joinpath(LOG_DIR, "bitmasks.jld")
save(LOG_DIR, "a", a)
