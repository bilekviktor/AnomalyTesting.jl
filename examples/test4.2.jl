using ToyProblems, DistributionsAD, SumProductTransform, Unitary, Flux, Setfield
using Distributions, Statistics, MLDataPattern, EvalMetrics
using Flux:throttle
using SumProductTransform: fit!, maptree
using ToyProblems: flower2
using SumProductTransform: ScaleShift, SVDDense
using DistributionsAD: TuringMvNormal
using Zygote
using DelimitedFiles
include("..//src//AnomalyTools.jl")

using Plots
using CSV, DataFrames

# SPTN definition
function sptn(d, n, l)
	m = TransformationNode(ScaleShift(d),  TuringMvNormal(d,1f0))
	for i in 1:l
		m = SumNode([TransformationNode(SVDDense(d, identity, :butterfly), m)
					for i in 1:n])
	end
	return(m)
end

function l(i, x, rho)
	if i == 1
		return max(0.0, rho - x)
	else
		return 1/2*min(x^2, rho)
	end
end
plot(-10:0.1:10, l.(-1, -10:0.1:10, 1.0))


struct AnomModel{M, R}
	m::M
	rho::R
end

Flux.@functor AnomModel

function Distributions.logpdf(m::AnomModel, x::AbstractArray{T}) where {T}
	return logpdf(m.m, x)
end

Zygote.@nograd ngpdf(m, x) = exp.(logpdf(m, x))

function anomloss(m::AnomModel, x::AbstractArray{T}, l) where {T}
	f = exp.(logpdf(m, x))
	return mean(l.(1, f, m.rho) + l.(-1, f, m.rho)./f)
end


m = sptn(2, 9, 1)
mm = AnomModel(m, 1.0)
x_train = flower2(1000, npetals=9)


x = RandomBatches(x_train, 100, 100)

loss(x) = anomloss(mm, x, l)

cb = function()
	println("loss: ", loss(x_train))
end

ps = Flux.params(mm)

opt = ADAM(0.1)
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

prplot(targets, scores_ml)

fpr = 0.05
tresh_line = [fpr 0.0; fpr 1.0]
rocplot(targets, scores_ml)
plot!(tresh_line[:, 1], tresh_line[:, 2], linestyle =:dashdot, linecolor=:grey, label="fpr treshhold")

θ_fpr = threshold_at_fpr(targets, scores_ml, 0.05)

anom, norms = tresh_lkl(m, x_noised, θ_fpr)
scatter(norms[1, :], norms[2, :])
scatter!(anom[1, :], anom[2, :])
