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
m = sptn(8, 9, 2)
M_DIR = normpath(@__DIR__)
NORMALTXT_PATH = normpath(M_DIR, "normal.txt")
x_train = transpose(readdlm(NORMALTXT_PATH))
x = RandomBatches(x_train, 1000, 100)
#scatter(x_train[1, :], x_train[2, :])

loss(x) = -mean(logpdf(m, x))

cb = function()
	println(loss(x_train))
end

ps = Flux.params(m)

opt = ADAM(0.1)
Flux.Optimise.train!(x -> loss(getobs(x)), ps, x, opt, cb=cb)


x_anom = rand(8, 1000)
x_noised = hcat(x_train, x_anom)
#scatter(x_noised[1, :], x_noised[2, :])

targets = vcat(ones(752), zeros(1000)) .!= 1
scores_8ml = -logpdf(m, x_noised)

fpr = 0.05
binary_eval_report(targets, scores_8ml)

tresh_line = [fpr 0.0; fpr 1.0]
prplot(targets, scores_8ml)
rocplot(targets, scores_8ml)
plot!(tresh_line[:, 1], tresh_line[:, 2], linestyle =:dashdot, linecolor=:grey, label="fpr treshhold")

# Classic quantile loss
m = sptn(8, 9, 2)
M_DIR = normpath(@__DIR__)
NORMALTXT_PATH = normpath(M_DIR, "normal.txt")
x_train = transpose(readdlm(NORMALTXT_PATH))
x = RandomBatches(x_train, 1000, 100)
#scatter(x_train[1, :], x_train[2, :])

loss(x) = -quantile_loss(m, x, 0.05, range=(1, 1))

cb = function()
	println(loss(x_train))
end

ps = Flux.params(m)

opt = ADAM(0.1)
Flux.Optimise.train!(x -> loss(getobs(x)), ps, x, opt, cb=cb)


x_anom = rand(8, 1000)
x_noised = hcat(x_train, x_anom)
#scatter(x_noised[1, :], x_noised[2, :])

targets = vcat(ones(752), zeros(1000)) .!= 1
scores_8q = -logpdf(m, x_noised)

fpr = 0.05
binary_eval_report(targets, scores_8q)

tresh_line = [fpr 0.0; fpr 1.0]
prplot(targets, scores_8q)
rocplot(targets, scores_8q)
plot!(tresh_line[:, 1], tresh_line[:, 2], linestyle =:dashdot, linecolor=:grey, label="fpr treshhold")


tresh_line = [fpr 0.0; fpr 1.0]
rocplot([targets, targets], [scores_ml, scores_q], label = ["ml" "quantile"])
plot!(tresh_line[:, 1], tresh_line[:, 2], linestyle =:dashdot, linecolor=:grey, label="fpr treshhold")
