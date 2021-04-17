using ILStrudel
using LogicCircuits
using ProbabilisticCircuits

t = twenty_dataset_names[1:end-1]

for d in t
	println(d)
	generate_pmi_bagging_stats(d; num_iters=300)
end
