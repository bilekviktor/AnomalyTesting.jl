using InvertedIndices
using Statistics
using Zygote
using Plots
using EvalMetrics

function tresh_lkl(m, x, θ, an_score=-1)
    loglhl = an_score*logpdf(m, x)
    an_index = findall(loglhl .< θ)
    return x[:, an_index], x[:, Not(an_index)]
end

function quantile_tresh(x, quan)
    sx = sort(x)
    n = length(x)
    quan_nmb = Int16(ceil(quan * n))
    return sx[quan_nmb]
end

function quantile_loss(m, x, quan; range::Tuple{Int, Int} = (0, 0))
    lkl = logpdf(m, x)
    slkl = sort(lkl)
    n = length(slkl)
    quan_nmb = Int16(ceil(quan * n))
    return mean(slkl[quan_nmb-range[1]:quan_nmb+range[2]])
end

function quantile_take(transformation, batch, qunatile, percentage=0.02)
    score = logpdf(transformation, batch)
    n = nobs(batch)
    q = Int(ceil(qunatile*n))
    perms = sortperm(score)
    range = Int(ceil((nobs(batch)*percentage)/2))
    indxs = perms[q-range:q+range]
    return batch[:, indxs]
end
Zygote.@nograd quantile_take

function right_quantile_take(transformation, batch, qunatile, percentage=0.02)
    score = logpdf(transformation, batch)
    n = nobs(batch)
    q = Int(ceil(qunatile*n))
    perms = sortperm(score)
    range = Int(ceil((nobs(batch)*percentage)))
    indxs = perms[q:q+range]
    return batch[:, indxs]
end
Zygote.@nograd right_quantile_take

function lkl_quantile(transformation, batch, percentage=(0.0, 1.0))
    score = logpdf(transformation, batch)
    n = nobs(batch)
    l, r = percentage
    if l <= 0.0
        lq = 1
    else
        lq = Int(ceil(l*n))
    end
    if r >= 1.0
        rq = n
    else
        rq = Int(ceil(r*n))
    end
    perms = sortperm(score)
    indxs = perms[lq:rq]
    return batch[:, indxs]
end
Zygote.@nograd lkl_quantile

function likelihood_sort(m, x)
    score = logpdf(m, x)
    n = nobs(x)
    perms = sortperm(score)
    return x[:, perms]
end

function norm_sort(m, x)
    y = m(x)
    r = [norm(y[:, i]) for i=1:size(x, 2)]
    perms = sortperm(r)
    return x[:, perms]
end

Zygote.@nograd bern_coef(n, k, q) = binomial(n-1, k-1)*q^(k-1)*(1-q)^(n-k)

function bernstein_est(x, q)    #(1-q) qunatile
    x_q = 0.0
    n = length(x)
    for k in 1:n
        x_q += bern_coef(n, k, q)*x[k]
    end
    return x_q
end

function model_radius(m, x)
    y = m(x)
    r = [norm(y[:, i]) for i=1:size(x, 2)]
    return r
end

function sumlogdet(m::SumNode, x::AbstractMatrix)
	lkl = logpdf(m, x) .- logpdf(TuringMvNormal(2,1f0), x)
end

function two_square_to_circ(z)
    x = z[1, :]
    y = z[2, :]
    cx = x .* sqrt.(1 .- y.^2/2)
    cy = y .* sqrt.(1 .- x.^2/2)
    return Array(hcat(cx, cy)')
end

function two_ball_rand(m)
    sq = 2*(rand(2, m) .- 0.5)
    return two_square_to_circ(sq)
end


function flow_volume(model, x, q, m)
    n = size(x, 1)
    r = model_radius(model, x)
    Rq = bernstein_est(r, q)
    eps = two_ball_rand(m)
    tmp, w = model((Rq .* eps, 0.0))
    logvol = log(4*pi/m) + n*log(Rq) + log(sum(exp.(w)))
end

function target_score(data, anoms, model, fpr)
	ndat = nobs(data)
	nanom = nobs(anoms)
	targets = vcat(ones(ndat), zeros(nanom)) .!= 1
	noised_d = hcat(data, anoms)
	scores = -logpdf(model, noised_d)
	return (targets, scores)
end

function roc_comparison(array_targets, array_scores, fpr=0.0, train_perc=0.0)
	pl = rocplot(array_targets, array_scores)
	if fpr == 0.0
		return pl
	elseif train_perc == 0.0
		tresh_line = [fpr 0.0; fpr 1.0]
		plot!(tresh_line[:, 1], tresh_line[:, 2], linestyle =:dashdot, linecolor=:grey, label="fpr treshhold")
        return pl
    else
        tresh_line = [fpr 0.0; fpr 1.0]
        l_line = [fpr-train_perc/2 0.0; fpr-train_perc/2 1.0]
        r_line = [fpr+train_perc/2 0.0; fpr+train_perc/2 1.0]
        plot!(tresh_line[:, 1], tresh_line[:, 2], linestyle =:dashdot, linecolor=:grey, label="fpr treshhold")
        plot!(l_line[:, 1], l_line[:, 2], linestyle =:dot, linecolor=:grey, label="train range")
        plot!(r_line[:, 1], r_line[:, 2], linestyle =:dot, linecolor=:grey, label="")
        return pl
    end
end
