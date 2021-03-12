
function bagging(train_x, valid_x, test_x, num_bagging_components; num_em_components=3,
		 maxiter=100, pseudocount=1.0)
	bags = []
	num_examples = size(train_x)[1]
	for i in 1:num_bagging_components
		ids = rand(1:num_examples, num_examples)
		train_x_ite = copy(train_x[ids, :])
		# Learn boosting_em ensemble
		m = boosting(train_x_ite, valid_x, test_x, num_em_components; pseudocount=pseudocount, maxiter=maxiter)
		push!(bags, m)
	end
	return bags
end

