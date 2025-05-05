using Gadfly, Cairo, Fontconfig, Distributions, Mamba

"""
    plotConvergenceCriterion(MCKL)

Plot evolution of the KL divergence over CAVI epochs.

# Arguments
- `MCKL::DenseVector`: Values of the KL divergence for each epoch.
- `saveFolder::String`: Folder where to save the fig.
"""
function plotConvergenceCriterion(MCKL::DenseVector; saveFolder::String)
    
    set_default_plot_size(15cm ,10cm)

    n_mckl = length(MCKL);

    p = plot(
        layer(x=1:n_mckl, y=MCKL, Geom.line),
        layer(x=1:n_mckl, y=MCKL, Geom.point, shape=[Shape.cross], Theme(default_color="red")),
        Theme(background_color="white"),
        Guide.title("Critère de convergence"),
        Guide.xlabel("Epoch"),
        Guide.ylabel("Divergence KL"),
    )

    draw(SVG("$saveFolder/mckl.svg"), p)
end


"""
    plotTraceCAVI(trace, name)

Plot evolution of the KL divergence over CAVI epochs.

# Arguments
- `trace::DenseVector`: Trace of the parameter.
- `name::String`: Name of the parameter.
- `saveFolder::String`: Folder where to save the fig.
"""
function plotTraceCAVI(trace::DenseVector, name::String; saveFolder::String)
    
    set_default_plot_size(15cm ,10cm)

    n_trace = length(trace);

    p = plot(
        layer(x=1:n_trace, y=trace, Geom.line),
        layer(x=1:n_trace, y=trace, Geom.point, shape=[Shape.cross], Theme(default_color="red")),
        Theme(background_color="white"),
        Guide.title("Trace CAVI de $name"),
        Guide.xlabel("Itération"),
        Guide.ylabel("Valeur"),
    )

    draw(SVG("$saveFolder/$(name)_cavi_trace.svg"), p)
end


"""
    plotTraceMCMC(chain, name)

Plot evolution of the KL divergence over CAVI epochs.

# Arguments
- `chain::Mamba.Chains`: Traces of all parameters.
- `name::String`: Name of the parameter.
- `saveFolder::String`: Folder where to save the fig.
"""
function plotTraceMCMC(chain::Mamba.Chains, name::String; saveFolder::String)
    
    set_default_plot_size(15cm ,10cm)

    trace = chain[:, name, 1].value;
    n_trace = length(trace);

    p = plot(
        layer(x=1:n_trace, y=trace, Geom.line),
        Theme(background_color="white"),
        Guide.title("Trace MCMC de $name"),
        Guide.xlabel("Itération"),
        Guide.ylabel("Valeur"),
    )

    draw(SVG("$saveFolder/$(name)_mcmc_trace.svg"), p)
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
    saveFolder::String,
)

    set_default_plot_size(20cm, 31cm)

    # x = 0:.0001:.1;
    x = 35:.001:45;

    marginal = buildCellCAVImarginal(numCell, 1, caviRes=caviRes);
    mcmcSample = mcmcChain[:, "μ$numCell", 1].value[warmingSize:end];

    p1 = plot(
        layer(x=x, y=pdf.(marginal, x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSample, Geom.histogram(density=true)),
        Guide.manual_color_key("", ["MCMC", "Approximation"], ["deepskyblue", "red"]),
        Theme(background_color="white"),
        Guide.title("SQAVI vs MCMC pour μ$numCell"),
        Guide.xlabel("mu"),
        Guide.ylabel("Densité"),
    );

    # x = -10:.01:0;
    x = 0:.01:2;
    
    marginal = buildCellCAVImarginal(numCell, 2, caviRes=caviRes);
    mcmcSample = mcmcChain[:, "ϕ$numCell", 1].value[warmingSize:end];
    
    p2 = plot(
        layer(x=x, y=pdf.(marginal, x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSample, Geom.histogram(density=true)),
        Guide.manual_color_key("", ["MCMC", "Approximation"], ["deepskyblue", "red"]),
        Theme(background_color="white"),
        Guide.title("SQAVI vs MCMC pour ϕ$numCell"),
        Guide.xlabel("phi"),
        Guide.ylabel("Densité"),
    );
        
    # x = 0:.0001:.15;
    x = 0.04:.0001:.06;

    marginal = caviRes.approxMarginals[M+1];
    mcmcSample = mcmcChain[:, "ξ", 1].value[warmingSize:end];

    p3 = plot(
        layer(x=x, y=pdf.(marginal, x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSample, Geom.histogram(density=true)),
        Guide.manual_color_key("", ["MCMC", "Approximation"], ["deepskyblue", "red"]),
        Theme(background_color="white"),
        Guide.title("SQAVI vs MCMC pour xi"),
        Guide.xlabel("xi"),
        Guide.ylabel("Densité"),
    );

    # x = 3*10^4:1:5*10^4;
    x = 0.7:.001:1.3;

    marginal = caviRes.approxMarginals[M+2];
    mcmcSample = mcmcChain[:, "κᵤ", 1].value[warmingSize:end];

    p4 = plot(
        layer(x=x, y=pdf.(marginal, x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSample, Geom.histogram(density=true)),
        Guide.manual_color_key("", ["MCMC", "Approximation"], ["deepskyblue", "red"]),
        Theme(background_color="white"),
        Guide.title("SQAVI vs MCMC pour kappa_u"),
        Guide.xlabel("kappa_u"),
        Guide.ylabel("Densité"),
    );
    
    # x = 0:.1:300;
    x = 8:.01:12;

    marginal = caviRes.approxMarginals[M+3];
    mcmcSample = mcmcChain[:, "κᵥ", 1].value[warmingSize:end];

    p5 = plot(
        layer(x=x, y=pdf.(marginal, x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSample, Geom.histogram(density=true)),
        Guide.manual_color_key("", ["MCMC", "Approximation"], ["deepskyblue", "red"]),
        Theme(background_color="white"),
        Guide.title("SQAVI vs MCMC pour kappa_v"),
        Guide.xlabel("kappa_v"),
        Guide.ylabel("Densité"),
    );

    p = vstack(p1, p2, p3, p4, p5)

    draw(SVG("$saveFolder/cavi_vs_mcmc_$numCell.svg"), p)
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
