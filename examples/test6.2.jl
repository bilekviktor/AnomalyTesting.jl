using ToyProblems, DistributionsAD, SumProductTransform, Unitary, Flux, Setfield
using Distributions, Statistics, MLDataPattern, EvalMetrics
using Flux:throttle
using SumProductTransform: fit!, maptree
using ToyProblems: flower2
using SumProductTransform: ScaleShift, SVDDense
using DistributionsAD: TuringMvNormal
using DelimitedFiles
include("..//src//AnomalyTools.jl")

using Plots


# SPTN definition
function sptn(d, n, l)
	m = TransformationNode(ScaleShift(d),  TuringMvNormal(d,1f0))
	for i in 1:l
		m = SumNode([TransformationNode(SVDDense(d, identity, :butterfly), m)
					for i in 1:n])
	end
	return(m)
end

# Classic max-likelihood loss function
# data
m1 = sptn(2, 9, 2)
x_train = flower2(10000, npetals=9)
x_val = flower2(1000, npetals=9)

# prep and training
loss(x) = -mean(logpdf(m1, x))

cb = function()
    qx_val = lkl_quantile(m1, x_val, (0.05, 0.1))
	println("Loss: ", loss(qx_val))
    println("Trainsize: ", length(findall(data.quantileInf.==1)))
end

data = QuantileData(x_train, (0.05, 0.1), 1000)

ps = Flux.params(m1)

opt = ADAM(0.01)
qunatile_train!(loss, ps, data, 1000, m1, opt; cb = cb)

#heatmap
xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-10.0:0.1:10, -10.0:0.1:10)])
res = reshape(logpdf(m1, xx), 201, 201)
Plots.heatmap(exp.(res))

#gen anoms and test data
x_test = flower2(100000, npetals=9)
x_anom = 10 .* (rand(2, 5000) .- 0.5)
fpr = 0.05

# evaluation
targets1, scores1 = target_score(x_test, x_anom, m1, fpr)
binary_eval_report(targets1, scores1)

prplot(targets1, scores1)
roc_comparison(targets1, scores1, fpr)
