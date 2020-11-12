#using FfjordFlow
using ToyProblems, DistributionsAD, SumProductTransform, Unitary, Flux, Setfield
using Distributions, Statistics, MLDataPattern
using Flux:throttle
using SumProductTransform: fit!, maptree
using ToyProblems: flower2
using SumProductTransform: ScaleShift, SVDDense
using DistributionsAD: TuringMvNormal
include("..//src//AnomalyTools.jl")

using Plots

function sptn(d, n, l)
	m = TransformationNode(ScaleShift(d),  TuringMvNormal(d,1f0))
	for i in 1:l
		m = SumNode([TransformationNode(SVDDense(2, identity, :butterfly), m)
					for i in 1:n])
	end
	return(m)
end

#------anomaly detection from lkl estimate--------#
m = sptn(2, 9, 1)

x_n = flower2(1000, npetals=9)
x_a = randn(2, 10)
scatter(x_n[1, :], x_n[2, :])
scatter!(x_a[1, :], x_a[2, :])
x = shuffleobs(hcat(x_n, x_a))
#scatter(x[1, :], x[2, :])
_data = Iterators.repeated((), 1000)

loss() = -mean(logpdf(m, x))

cbb = function()
	println(loss())
end

ps = Flux.params(m)

opt = ADAM(0.1)
Flux.Optimise.train!(() -> loss(), ps, _data, opt, cb = cbb)

xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-10.0:0.1:10, -10.0:0.1:10)])
res = reshape(logpdf(m, xx), 201, 201)
Plots.heatmap(exp.(res))

lkl = logpdf(m, x)
θ = quantile_tresh(lkl, 0.01)

anom, norms = tresh_lkl(m, x, θ)
scatter(norms[1, :], norms[2, :])
scatter!(anom[1, :], anom[2, :])

#----------anomaly detection with quantile loss------------#
m = sptn(2, 9, 1)

x_n = flower2(1000, npetals=9)
x_a = randn(2, 10)
scatter(x_n[1, :], x_n[2, :])
scatter!(x_a[1, :], x_a[2, :])
x = shuffleobs(hcat(x_n, x_a))
#scatter(x[1, :], x[2, :])
_data = Iterators.repeated((), 1000)

loss() = -quantile_loss(m, x, 0.01, range=(0, 40))

cbb = function()
	println(loss())
end

ps = Flux.params(m)

opt = ADAM(0.1)
Flux.Optimise.train!(() -> loss(), ps, _data, opt, cb = cbb)

xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-10.0:0.1:10, -10.0:0.1:10)])
res = reshape(logpdf(m, xx), 201, 201)
Plots.heatmap(exp.(res))

lkl = logpdf(m, x)
θ = quantile_tresh(lkl, 0.01)

anom, norms = tresh_lkl(m, x, θ)
scatter(norms[1, :], norms[2, :])
scatter!(anom[1, :], anom[2, :])
