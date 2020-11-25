using InvertedIndices
using Statistics


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
