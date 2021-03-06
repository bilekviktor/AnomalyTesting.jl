using ToyProblems, DistributionsAD, SumProductTransform, Unitary, Flux, Setfield
using Distributions, Statistics, MLDataPattern, EvalMetrics
using Flux:throttle
using SumProductTransform: fit!, maptree
using ToyProblems: flower2
using SumProductTransform: ScaleShift, SVDDense
using DistributionsAD: TuringMvNormal
using FfjordFlow
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
m = sptn(2, 9, 2)
x_train = flower2(100000, npetals=9)
x = RandomBatches(x_train, 1000, 1000)
#scatter(x_train[1, :], x_train[2, :])

loss(x) = -mean(logpdf(m, x))

cb = function()
	println(loss(x_train))
end

ps = Flux.params(m)

opt = ADAM(0.01)
Flux.Optimise.train!(x -> loss(getobs(x)), ps, x, opt, cb=cb)

xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-10.0:0.1:10, -10.0:0.1:10)])
res = reshape(logpdf(m, xx), 201, 201)
Plots.heatmap(exp.(res))

x_anom = 10 .* (rand(2, 10000) .- 0.5)
x_noised = hcat(x_train, x_anom)
#scatter(x_noised[1, :], x_noised[2, :])

targets = vcat(ones(99999), zeros(10000)) .!= 1
scores_ml = -logpdf(m, x_noised)

binary_eval_report(targets, scores_ml)

fpr = 0.05

prplot(targets, scores_ml)
tresh_line = [fpr 0.0; fpr 1.0]
rocplot(targets, scores_ml)
plot!(tresh_line[:, 1], tresh_line[:, 2], linestyle =:dashdot, linecolor=:grey, label="fpr treshhold")

θ_fpr = threshold_at_fpr(targets, scores_ml, fpr)

anom, norms = tresh_lkl(m, x_noised, θ_fpr)
scatter(norms[1, :], norms[2, :])
scatter!(anom[1, :], anom[2, :])

# Quantile loss function
m = sptn(2, 9, 2)
x_train = flower2(100000, npetals=9)
x = RandomBatches(x_train, 1000, 100)
#scatter(x_train[1, :], x_train[2, :])

qx = (quantile_take(m, batch, 0.05) for batch in x)

loss(x) = -mean(logpdf(m, x))

cb = function()
	println(loss(x_train))
end

ps = Flux.params(m)

opt = ADAM(0.01)
Flux.Optimise.train!(x -> loss(getobs(x)), ps, qx, opt, cb=cb)

xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-10.0:0.1:10, -10.0:0.1:10)])
res = reshape(logpdf(m, xx), 201, 201)
Plots.heatmap(exp.(res))

x_anom = 10 .* (rand(2, 10000) .- 0.5)
x_noised = hcat(x_train, x_anom)
#scatter(x_noised[1, :], x_noised[2, :])

targets = vcat(ones(99999), zeros(10000)) .!= 1
scores_q = -logpdf(m, x_noised)

binary_eval_report(targets, scores_q)
fpr = 0.05

prplot(targets, scores_q)

tresh_line = [fpr 0.0; fpr 1.0]
rocplot(targets, scores_ml)
plot!(tresh_line[:, 1], tresh_line[:, 2], linestyle =:dashdot, linecolor=:grey, label="fpr treshhold")

θ_fpr = threshold_at_fpr(targets, scores_q, fpr)

anom, norms = tresh_lkl(m, x_noised, θ_fpr)
scatter(norms[1, :], norms[2, :])
scatter!(anom[1, :], anom[2, :])

#Comparison of qunatile loss nad mlh loss
tresh_line = [fpr 0.0; fpr 1.0]
rocplot([targets, targets], [scores_ml, scores_q], label = ["ml" "quantile"])
plot!(tresh_line[:, 1], tresh_line[:, 2], linestyle =:dashdot, linecolor=:grey, label="fpr treshhold")
