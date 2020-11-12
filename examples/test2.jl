using ToyProblems, DistributionsAD, SumProductTransform, Unitary, Flux, Setfield
using Distributions, Statistics, MLDataPattern, EvalMetrics
using Flux:throttle
using SumProductTransform: fit!, maptree
using ToyProblems: flower2
using SumProductTransform: ScaleShift, SVDDense
using DistributionsAD: TuringMvNormal
include("..//src//AnomalyTools.jl")

using Plots

# SPTN definition
function sptn(d, n, l)
	m = TransformationNode(ScaleShift(d),  TuringMvNormal(d,1f0))
	for i in 1:l
		m = SumNode([TransformationNode(SVDDense(2, identity, :butterfly), m)
					for i in 1:n])
	end
	return(m)
end

# Playing with EvalMetrics
targets = rand(0:1, 100)
scores = rand(100)

binary_eval_report(targets, scores)

prplot(targets, scores)
rocplot(targets, scores)

predicts = scores .>= 0.6
cm0 = ConfusionMatrix(targets, scores, 0.0)
cm1 = ConfusionMatrix(targets, scores, 1.0)

# Classic max-likelihood loss function
m = sptn(2, 9, 2)
x_train = flower2(100000, npetals=9)
x = RandomBatches(x_train, 100, 100)
#scatter(x_train[1, :], x_train[2, :])

loss(x) = -mean(logpdf(m, x))

cb = function()
	println(loss(x_train))
end

ps = Flux.params(m)

opt = ADAM(0.1)
Flux.Optimise.train!(x -> loss(getobs(x)), ps, x, opt, cb=cb)

xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-10.0:0.1:10, -10.0:0.1:10)])
res = reshape(logpdf(m, xx), 201, 201)
Plots.heatmap(exp.(res))

x_anom = 10 .* (rand(2, 10000) .- 0.5)
x_noised = hcat(x_train, x_anom)
#scatter(x_noised[1, :], x_noised[2, :])

targets = vcat(ones(99999), zeros(10000)) .== 1
scores_ml = logpdf(m, x_noised)

binary_eval_report(targets, scores_ml)

prplot(targets, scores_ml)
rocplot(targets, scores_ml)

θ_fnr = threshold_at_fnr(targets, scores_ml, 0.01)

anom, norms = tresh_lkl(m, x_noised, θ_fnr)
scatter(norms[1, :], norms[2, :])
scatter!(anom[1, :], anom[2, :])

# Quantile loss function
m = sptn(2, 9, 2)
x_train = flower2(100000, npetals=9)
x = RandomBatches(x_train, 100, 100)
#scatter(x_train[1, :], x_train[2, :])

loss(x) = -quantile_loss(m, x, 0.1, range=(0, 70))

cb = function()
	println(loss(x_train))
end

ps = Flux.params(m)

opt = ADAM(0.1)
Flux.Optimise.train!(x -> loss(getobs(x)), ps, x, opt, cb=cb)

xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-10.0:0.1:10, -10.0:0.1:10)])
res = reshape(logpdf(m, xx), 201, 201)
Plots.heatmap(exp.(res))

x_anom = 10 .* (rand(2, 10000) .- 0.5)
x_noised = hcat(x_train, x_anom)
#scatter(x_noised[1, :], x_noised[2, :])

targets = vcat(ones(99999), zeros(10000)) .== 1
scores_q = logpdf(m, x_noised)

binary_eval_report(targets, scores_q)

prplot(targets, scores_q)
rocplot(targets, scores_q)

θ_fnr = threshold_at_fnr(targets, scores_q, 0.01)

anom, norms = tresh_lkl(m, x_noised, θ_fnr)
scatter(norms[1, :], norms[2, :])
scatter!(anom[1, :], anom[2, :])


rocplot([targets, targets], [scores_ml, scores_q], label = ["ml" "quantile"])
# Classic max-likelihood loss function - training with anomalies
m = sptn(2, 9, 2)
x_train = flower2(100000, npetals=9)
x_anom = 10 .* (rand(2, 10000) .- 0.5)
x = RandomBatches(shuffleobs(hcat(x_train, x_anom)), 100, 100)
#scatter(x_train[1, :], x_train[2, :])

loss(x) = -mean(logpdf(m, x))

cb = function()
	println(loss(x_train))
end

ps = Flux.params(m)

opt = ADAM(0.1)
Flux.Optimise.train!(x -> loss(getobs(x)), ps, x, opt, cb=cb)

xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-10.0:0.1:10, -10.0:0.1:10)])
res = reshape(logpdf(m, xx), 201, 201)
Plots.heatmap(exp.(res))

x_noised = hcat(x_train, x_anom)
#scatter(x_noised[1, :], x_noised[2, :])

targets = vcat(ones(99999), zeros(10000)) .== 1
scores_anml = logpdf(m, x_noised)

binary_eval_report(targets, scores_anml)

prplot(targets, scores_anml)
rocplot(targets, scores_anml)

θ_fnr = threshold_at_fnr(targets, scores_anml, 0.01)

anom, norms = tresh_lkl(m, x_noised, θ_fnr)
scatter(norms[1, :], norms[2, :])
scatter!(anom[1, :], anom[2, :])

# Quantile loss function - training with anomalies
m = sptn(2, 9, 2)
x_train = flower2(100000, npetals=9)
x_anom = 10 .* (rand(2, 10000) .- 0.5)
x = RandomBatches(shuffleobs(hcat(x_train, x_anom)), 100, 100)
#scatter(x_train[1, :], x_train[2, :])

loss(x) = -quantile_loss(m, x, 0.1, range=(0, 70))

cb = function()
	println(loss(x_train))
end

ps = Flux.params(m)

opt = ADAM(0.1)
Flux.Optimise.train!(x -> loss(getobs(x)), ps, x, opt, cb=cb)

xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-10.0:0.1:10, -10.0:0.1:10)])
res = reshape(logpdf(m, xx), 201, 201)
Plots.heatmap(exp.(res))

x_noised = hcat(x_train, x_anom)
#scatter(x_noised[1, :], x_noised[2, :])

targets = vcat(ones(99999), zeros(10000)) .== 1
scores_anq = logpdf(m, x_noised)

binary_eval_report(targets, scores_anq)

prplot(targets, scores_anq)
rocplot(targets, scores_anq)

θ_fnr = threshold_at_fnr(targets, scores_anq, 0.01)

anom, norms = tresh_lkl(m, x_noised, θ_fnr)
scatter(norms[1, :], norms[2, :])
scatter!(anom[1, :], anom[2, :])

rocplot([targets, targets], [scores_anml, scores_anq], label = ["ml" "quantile"])
