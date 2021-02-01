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

# VAE

M_DIR = normpath(@__DIR__)
NORMALTXT_PATH = normpath(M_DIR, "normal.txt")
x = transpose(readdlm(NORMALTXT_PATH))

d = Chain(Dense(8, 32, tanh), Dense(32, 16))
e = Chain(Dense(8, 32, tanh), Dense(32, 16))

function p(x, z)
	tmp = d(z)
	mu = tmp[1:8]
	sig = tmp[9:16]
	n = TuringDiagMvNormal(mu, sig.^2)
	return logpdf(n, x)
end

function q(z, x)
	tmp = e(x)
	mu = tmp[1:8]
	sig = tmp[9:16]
	n = TuringDiagMvNormal(mu, sig.^2)
	return logpdf(n, z)
end

function KLdiv(p, q)
	sum((exp.(p) .* p) .- (exp.(p) .* q))
end


function loss(x)
	z = randn(8, 752)
	pp=[p(x[:, i], z[:, i]) for i in 1:752]
	qq=[q(z[:, i], x[:, i]) for i in 1:752]
	l = KLdiv(qq, pp) - mean(pp)
end

loss() = loss(x)

ps = Flux.params(d, e)
opt = ADAM(0.0001)

cb = function()
	println(loss())
end
_data = Iterators.repeated((), 100)

Flux.Optimise.train!(() -> loss(), ps, _data, opt, cb=cb)

xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-10.0:0.1:10, -10.0:0.1:10)])
zz = rand(2, 40401)
res = reshape([p(xx[:, i], zz[:, i]) for i in 1:40401], 201, 201)
Plots.heatmap(exp.(res))




# GAN
x = flower2(10000)
z = randn(size(x))
n = 20
generator = Chain(Dense(2, n, tanh), Dense(n, n, tanh), Dense(n, 2))
discriminator = Chain(Dense(2, n, tanh), Dense(n, n, tanh), Dense(n, 1, σ))

function loss_GAN(x)
	#z = randn(size(x))
	l = mean(log.(discriminator(x))) + mean(log.(1 .- discriminator(generator(z))))
end

ps_gen = Flux.params(generator)
ps_dis = Flux.params(discriminator)

opt = GAN_ADAM(0.1)
_data = Iterators.repeated((), 100)

cb = function()
	println(loss_GAN(x))
	# res = reshape(discriminator(xx), 201, 201)
	# display(Plots.heatmap(res))
end

GAN_train!(() -> loss_GAN(x), ps_gen, ps_dis, _data, opt, cb=cb)

xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-10.0:0.1:10, -10.0:0.1:10)])
res = reshape(discriminator(xx), 201, 201)
Plots.heatmap(res)

x_train = x
x_anom = 10 .* (rand(2, 1000) .- 0.5)
x_noised = hcat(x_train, x_anom)
#scatter(x_noised[1, :], x_noised[2, :])

targets = vcat(ones(1000), zeros(1000)) .!= 1
scores_GAN = transpose(discriminator(x_noised))

binary_eval_report(targets, scores_GAN)

prplot(targets, scores_q)
rocplot(targets, scores_q)

fpr = 0.05
θ_fpr = threshold_at_fpr(targets, scores_q, fpr)

anom, norms = tresh_lkl(m, x_noised, θ_fpr)
scatter(norms[1, :], norms[2, :])
scatter!(anom[1, :], anom[2, :])


import Zygote: Params, gradient

mutable struct GAN_ADAM{T}
	a1::T
	a2::T
end

GAN_ADAM(α) = GAN_ADAM(ADAM(α), ADAM(-α))

function Flux.Optimise.update!(o::GAN_ADAM, x_gen, bx_gen, x_dis, bx_dis)
	Flux.Optimise.update!(o.a2, x_dis, bx_dis)
	Flux.Optimise.update!(o.a1, x_gen, bx_gen)
end

struct SkipException <: Exception end
function skip()
  throw(SkipException())
end


struct StopException <: Exception end
function stop()
  throw(StopException())
end

batchmemaybe(x) = tuple(x)
batchmemaybe(x::Tuple) = x


function GAN_train!(loss, ps_gen, ps_dis, data, opt; cb = () -> ())
	ps_gen = Params(ps_gen)
	ps_dis = Params(ps_dis)
	cb()
	@progress for d in data
		try
			gs_dis = gradient(ps_dis) do
		  		loss(batchmemaybe(d)...)
			end
		Flux.Optimise.update!(opt.a2, ps_dis, gs_dis)
		#println("DISC. TRAIN:")
		cb()
		catch ex
		  	if ex isa StopException
		    	break
		  	elseif ex isa SkipException
		    	continue
		  	else
		    	rethrow(ex)
		  	end
		end
	end
	@progress for d in data
		try
			gs_gen = gradient(ps_gen) do
				loss(batchmemaybe(d)...)
			end
		Flux.Optimise.update!(opt.a1, ps_gen, gs_gen)
		#println("GEN TRAIN")
		cb()
		catch ex
		  	if ex isa StopException
		    	break
		  	elseif ex isa SkipException
		    	continue
		  	else
		    	rethrow(ex)
		  	end
		end
	end
end

# Reconstruction
n=10
m = Chain(Dense(2, n, tanh), Dense(n, 2, tanh), Dense(2, 2))
x = flower2(1000)

ps = Flux.params(m)
loss(x) = sum(abs, x .- m(x))

_data = Iterators.repeated((), 1000)
opt = ADAM(0.1)

cb = function()
	println(loss(x))
end

Flux.Optimise.train!(() -> loss(x), ps, _data, opt, cb=cb)
s(x) = sum(abs, x .- m(x))

x_anom = 10 .* (rand(2, 1000) .- 0.5)
x_noised = hcat(x, x_anom)
#scatter(x_noised[1, :], x_noised[2, :])

targets = vcat(ones(1000), zeros(1000)) .!= 1
scores = [s(x_noised[:, i]) for i in 1:size(x_noised, 2)]

binary_eval_report(targets, scores)

prplot(targets, scores)
rocplot(targets, scores)

θ_fpr = threshold_at_fpr(targets, scores, 0.05)

ind = scores .< θ_fpr
nind = scores .>= θ_fpr
norms = x_noised[:, ind]
anom = x_noised[:, nind]
scatter(norms[1, :], norms[2, :])
scatter!(anom[1, :], anom[2, :])
