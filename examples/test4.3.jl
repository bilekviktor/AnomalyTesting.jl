using ToyProblems, DistributionsAD, SumProductTransform, Unitary, Flux, Setfield
using Distributions, Statistics, MLDataPattern, EvalMetrics
using Flux:throttle
using SumProductTransform: fit!, maptree
using ToyProblems: flower2
using SumProductTransform: ScaleShift, SVDDense
using DistributionsAD: TuringMvNormal
using DelimitedFiles
using FfjordFlow
using LinearAlgebra
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

# ONE-Flow
n = 20
m = Cnf(Chain(Dense(2, n, tanh), Dense(n, n, tanh), Dense(n, 2)), (0.0, 1.0))
x_train = flower2(100000, npetals=9)
x = RandomBatches(x_train, 10, 100)
sx = (norm_sort(m, batch) for batch in x)
fpr = 0.05

loss(x) = flow_volume(m, x, fpr, 100)
loss(norm_sort(m, getobs(x)[1]))

cb = function()
	println(exp(loss(norm_sort(m, getobs(x)[1]))))
end

ps = Flux.params(m)

opt = ADAM(0.01)
Flux.Optimise.train!(x -> loss(getobs(x)), ps, sx, opt, cb=cb)

mm = Cnf(m)

xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-10.0:0.1:10, -10.0:0.1:10)])
res = reshape(logpdf(m, xx), 201, 201)
Plots.heatmap(exp.(res))

y = m(x_train)
r = [norm(y[:, i]) for i=1:size(x_train, 2)]
sr = sortperm(r)
xv = x_train[:, sr]
xa = xv[:, 1:499]
xn = xv[:, 500:end]
scatter(xn[1, :], xn[2, :])
scatter(xa[1, :], xa[2, :])
