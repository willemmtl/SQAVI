using Gadfly, Cairo, Fontconfig, Distributions, Mamba

"""
    plotConvergenceCriterion(MCKL)

Plot evolution of the KL divergence over CAVI epochs.

# Arguments
- `MCKL::DenseVector`: Values of the KL divergence for each epoch.
"""
function plotConvergenceCriterion(MCKL::DenseVector)
    
    set_default_plot_size(15cm ,10cm)

    n_mckl = length(MCKL);

    plot(
        layer(x=1:n_mckl, y=MCKL, Geom.line),
        layer(x=1:n_mckl, y=MCKL, Geom.point, shape=[Shape.cross], Theme(default_color="red")),
        Theme(background_color="white"),
        Guide.title("Critère de convergence"),
        Guide.xlabel("Epoch"),
        Guide.ylabel("Divergence KL"),
    )
end


"""
    plotTraceCAVI(trace, name)

Plot evolution of the KL divergence over CAVI epochs.

# Arguments
- `trace::DenseVector`: Trace of the parameter.
- `name::String`: Name of the parameter.
"""
function plotTraceCAVI(trace::DenseVector, name::String)
    
    set_default_plot_size(15cm ,10cm)

    n_trace = length(trace);

    plot(
        layer(x=1:n_trace, y=trace, Geom.line),
        layer(x=1:n_trace, y=trace, Geom.point, shape=[Shape.cross], Theme(default_color="red")),
        Theme(background_color="white"),
        Guide.title("Trace CAVI de $name"),
        Guide.xlabel("Itération"),
        Guide.ylabel("Valeur"),
    )
end


"""
    plotTraceMCMC(chain, name)

Plot evolution of the KL divergence over CAVI epochs.

# Arguments
- `chain::Mamba.Chains`: Traces of all parameters.
- `name::String`: Name of the parameter.
"""
function plotTraceMCMC(chain::Mamba.Chains, name::String)
    
    set_default_plot_size(15cm ,10cm)

    trace = chain[:, name, 1].value;
    n_trace = length(trace);

    plot(
        layer(x=1:n_trace, y=trace, Geom.line),
        Theme(background_color="white"),
        Guide.title("Trace MCMC de $name"),
        Guide.xlabel("Itération"),
        Guide.ylabel("Valeur"),
    )
end


"""
    plotCAVIvsMCMC(numCell; caviRes, mcmcChain, warmingSize)

Plot approx marginals and histogram of MCMC samples for each parameter.

# Arguments
TBD
"""
function plotCAVIvsMCMC(
    numCell::Integer;
    caviRes::CAVIres,
    mcmcChain::Mamba.Chains, 
    warmingSize::Integer,
)

    set_default_plot_size(20cm, 31cm)

    x = 5:.001:15;

    marginal = buildCellCAVImarginal(numCell, 1, caviRes=caviRes);
    mcmcSample = mcmcChain[:, "μ$numCell", 1].value[warmingSize:end];

    p1 = plot(
        layer(x=x, y=pdf.(marginal, x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSample, Geom.histogram(density=true)),
        Theme(background_color="white"),
        Guide.title("CAVI vs MCMC pour mu"),
        Guide.xlabel("mu"),
        Guide.ylabel("Densité"),
    );

    x = -2:.001:2;

    marginal = buildCellCAVImarginal(numCell, 2, caviRes=caviRes);
    mcmcSample = mcmcChain[:, "ϕ$numCell", 1].value[warmingSize:end];

    p2 = plot(
        layer(x=x, y=pdf.(marginal, x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSample, Geom.histogram(density=true)),
        Theme(background_color="white"),
        Guide.title("CAVI vs MCMC pour phi"),
        Guide.xlabel("phi"),
        Guide.ylabel("Densité"),
    );

    x = .22:.0001:.27;

    marginal = caviRes.approxMarginals[M+1];
    mcmcSample = mcmcChain[:, "ξ", 1].value[warmingSize:end];

    p3 = plot(
        layer(x=x, y=pdf.(marginal, x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSample, Geom.histogram(density=true)),
        Theme(background_color="white"),
        Guide.title("CAVI vs MCMC pour xi"),
        Guide.xlabel("xi"),
        Guide.ylabel("Densité"),
    );

    x = .5:.001:1.5;

    marginal = caviRes.approxMarginals[M+2];
    mcmcSample = mcmcChain[:, "κᵤ", 1].value[warmingSize:end];

    p4 = plot(
        layer(x=x, y=pdf.(marginal, x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSample, Geom.histogram(density=true)),
        Theme(background_color="white"),
        Guide.title("CAVI vs MCMC pour kappa_u"),
        Guide.xlabel("kappa_u"),
        Guide.ylabel("Densité"),
    );

    x = 6:.001:14;

    marginal = caviRes.approxMarginals[M+3];
    mcmcSample = mcmcChain[:, "κᵥ", 1].value[warmingSize:end];

    p5 = plot(
        layer(x=x, y=pdf.(marginal, x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSample, Geom.histogram(density=true)),
        Theme(background_color="white"),
        Guide.title("CAVI vs MCMC pour kappa_v"),
        Guide.xlabel("kappa_v"),
        Guide.ylabel("Densité"),
    );

    vstack(p1, p2, p3, p4, p5)
end


"""
    buildCAVImarginal(numCell, paramNum; caviRes)

Build the marginal density of a parameter of a given cell from the result of the CAVI algorithm.

# Arguments
TBD
"""
function buildCellCAVImarginal(numCell::Integer, paramNum::Integer; caviRes::CAVIres)
    return Normal(
        params(caviRes.approxMarginals[numCell])[1][paramNum],
        sqrt(diag(params(caviRes.approxMarginals[numCell])[2])[paramNum])
    )
end
