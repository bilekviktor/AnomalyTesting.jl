using Zygote, Flux
using InvertedIndices
using MLDataPattern

function in_quantile(m, x, percentage=(0.0, 1.0))
    score = logpdf(m, x)
    n = nobs(x)
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
    q_indxs = perms[lq:rq]
    n_indxs = perms[Not(lq:rq)]
    return q_indxs, n_indxs
end

mutable struct QuantileData{T, W, N, I}
    x::T
    percentage::W
    batchSize::N
    quantileInf::I
end

function QuantileData(x, percentage, bacthSize)
    n = nobs(x)
    QuantileData(x, percentage, bacthSize, zeros(Int, n))
end

function qunatile_train!(loss, ps, data::QuantileData, iterations, model, opt; cb = () -> ())
    ps = Zygote.Params(ps)
    cb()
    x, percentage, batchSize = data.x, data.percentage,
                                            data.batchSize
    dataSize = nobs(x)
    for i in 1:iterations
        randomPerm = randobs(1:dataSize, batchSize)
        batch = x[:, randomPerm]
        q_indxs, n_indxs = in_quantile(model, batch, percentage)
        data.quantileInf[randomPerm[q_indxs]] .= 1
        data.quantileInf[randomPerm[n_indxs]] .= 0
        indxs = findall(data.quantileInf .== 1)
        qx = x[:, indxs]
        gs = gradient(() -> loss(qx), ps)
        Flux.Optimise.update!(opt, ps, gs)
        cb()
    end
end
