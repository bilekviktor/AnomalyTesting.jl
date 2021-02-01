using Distributions, ToyProblems, LinearAlgebra

function flower2_pdf(xy, npetals=8)
    x = xy[1, :]
    y = xy[2, :]
    theta = 1:npetals
    n = size(xy, 2)
    fpdf = zeros(n)
    xmu0 = 4.0
    ymu0 = 0.0
    xsig0 = 0.3
    ysig0 = 0.05
    for t in theta * (2pi/npetals)
        ct = cos(t)
        st = sin(t)
        A = [ct -st; st ct]
        mu = A*[xmu0, ymu0]
        sig = A*[xsig0 0; 0 ysig0]*transpose(A)
        d = MvNormal(mu, sig)
        fpdf .+= pdf(d, xy)
    end
    return fpdf./npetals
end

function atanh_pdf(x)
    d = Normal()
    if abs(x) < 1
        return pdf(d, atanh(x) + 1)*(1\(1-x^2))
    else
        return 0.0
    end
end


function petal_pdf(xy)
    x = xy[1, :]
    y = xy[2, :]
    dx1 = Normal(4.0, 0.05)
    dy = Normal(0.0, 0.3)
    pdfx1 = pdf.(dx1, x)
    pdfx2 = atanh_pdf.(x)
    pdfx = DSP.ifft(DSP.fft(pdfx1) .* DSP.fft(pdfx2))
    @show size(DSP.fft(pdfx1)), size(DSP.fft(pdfx2)), size(pdfx)
    pdfy = pdf.(dy, y)
    return pdfy .* pdfx
end
    


xx = reduce(hcat,[[v[1],v[2]] for v in Iterators.product(-10.0:0.1:10, -10.0:0.1:10)])
res = reshape(petal_pdf(xx), 201, 201)
Plots.heatmap((res))
