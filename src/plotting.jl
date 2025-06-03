using Gadfly, Cairo, Fontconfig, Distributions, Mamba, Measures


TICK_FONT_SIZE = 16pt;
SUBTICK_FONT_SIZE = 14pt;
LEGEND_FONT_SIZE = 16pt;

"""
    plotConvergenceCriterion(MCKL; saveFolder)

Plot evolution of the KL divergence over CAVI epochs.

# Arguments
- `MCKL::DenseVector`: Values of the KL divergence for each epoch.
- `saveFolder::String`: Folder where to save the fig.
"""
function plotConvergenceCriterion(MCKL::DenseVector; saveFolder::String)
    
    set_default_plot_size(15cm ,10cm)

    n_mckl = length(MCKL)-1;

    p = plot(
        layer(x=1:n_mckl, y=MCKL[2:end], Geom.line),
        layer(x=1:n_mckl, y=MCKL[2:end], Geom.point, shape=[Shape.cross], Theme(default_color="red")),
        Guide.Theme(
            background_color="white",
            major_label_font_size=TICK_FONT_SIZE,
            minor_label_font_size=SUBTICK_FONT_SIZE,
        ),
        # Guide.title("Critère de convergence"),
        Guide.xlabel("Époque"),
        Guide.ylabel("Divergence KL"),
    )

    draw(SVG("$saveFolder/plots/mckl.svg"), p)
end


"""
    plotTraceSQAVI(trace, name; saveFolder)

Plot evolution of the KL divergence over CAVI epochs.

# Arguments
- `trace::DenseVector`: Trace of the parameter.
- `name::String`: Name of the parameter.
- `saveFolder::String`: Folder where to save the fig.
"""
function plotTraceSQAVI(trace::DenseVector, name::String; saveFolder::String)
    
    set_default_plot_size(15cm ,10cm)

    n_trace = length(trace);

    p = plot(
        layer(x=0:n_trace-1, y=trace, Geom.line),
        layer(x=0:n_trace-1, y=trace, Geom.point, shape=[Shape.cross], Theme(default_color="red")),
        Guide.Theme(
            background_color="white",
            major_label_font_size=TICK_FONT_SIZE,
            minor_label_font_size=SUBTICK_FONT_SIZE,
        ),
        # Guide.title("Trace CAVI de $name"),
        Guide.xlabel("Itération"),
        Guide.ylabel("$name"),
    )

    draw(SVG("$saveFolder/plots/$(name)_sqavi_trace.svg"), p)
end


"""
    plotTraceMCMC(chain, name; saveFolder)

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
        Guide.Theme(
            background_color="white",
            major_label_font_size=TICK_FONT_SIZE,
            minor_label_font_size=SUBTICK_FONT_SIZE,
        ),
        # Guide.title("Trace MCMC de $name"),
        Guide.xlabel("Itération"),
        Guide.ylabel("Valeur"),
    )

    draw(SVG("$saveFolder/plots/$(name)_mcmc_trace.svg"), p)
end


function plotHistMCMC(chain::Mamba.Chains, name::String; warmingSize::Int, saveFolder::String)

    set_default_plot_size(15cm ,10cm)
    
    sample = chain[warmingSize:end, name, 1].value;
    
    p = plot(
        x=sample, Geom.histogram(density=true),
        Guide.Theme(
            background_color="white",
            major_label_font_size=TICK_FONT_SIZE,
            minor_label_font_size=SUBTICK_FONT_SIZE,
        ),
        Guide.xlabel("$name"),
        Guide.ylabel("Densité"),
    )

    draw(SVG("$saveFolder/plots/$(name)_mcmc_density.svg"), p)

end


"""
    plotSQAVIvsMCMC(numCell; caviRes, mcmcChain, warmingSize, saveFolder)

Plot approx marginals and histogram of MCMC samples for each parameter.

# Arguments
TBD
"""
function plotSQAVIvsMCMC(
    numCell::Integer;
    sqaviRes::SQAVIres,
    mcmcChain::Mamba.Chains, 
    warmingSize::Integer,
    saveFolder::String,
)

    M = size(sqaviRes.traces[:muMean], 1);
    mcmcIter = size(chain, 1);
    binCount = round(Int, sqrt(mcmcIter));

    set_default_plot_size(20cm, 40cm);

    x = 35:.001:55;

    marginal = buildCellCAVImarginal(numCell, 1, sqaviRes=sqaviRes);
    mcmcSample = mcmcChain[:, "μ$numCell", 1].value[warmingSize:end];

    p1 = plot(
        layer(x=x, y=pdf.(marginal, x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSample, Geom.histogram(density=true, bincount=binCount)),
        Guide.Theme(
            background_color="white",
            major_label_font_size=TICK_FONT_SIZE,
            minor_label_font_size=SUBTICK_FONT_SIZE,
            key_label_font_size=LEGEND_FONT_SIZE,
        ),
        # Guide.title("SQAVI vs MCMC pour μ$numCell"),
        Guide.xlabel("μ$numCell"),
        Guide.ylabel("Densité"),
    );

    x = 1.5:.0001:3;
    
    marginal = buildCellCAVImarginal(numCell, 2, sqaviRes=sqaviRes);
    mcmcSample = mcmcChain[:, "ϕ$numCell", 1].value[warmingSize:end];
    
    p2 = plot(
        layer(x=x, y=pdf.(marginal, x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSample, Geom.histogram(density=true, bincount=binCount)),
        Guide.Theme(
            background_color="white",
            major_label_font_size=TICK_FONT_SIZE,
            minor_label_font_size=SUBTICK_FONT_SIZE,
            key_label_font_size=LEGEND_FONT_SIZE,
        ),
        # Guide.title("SQAVI vs MCMC pour ϕ$numCell"),
        Guide.xlabel("ϕ$numCell"),
        Guide.ylabel("Densité"),
    );
        
    # x = 0:.0001:.15;
    x = .05:.00001:.07;

    marginal = sqaviRes.approxMarginals[M+1];
    mcmcSample = mcmcChain[:, "ξ", 1].value[warmingSize:end];

    p3 = plot(
        layer(x=x, y=pdf.(marginal, x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSample, Geom.histogram(density=true, bincount=binCount)),
        Guide.Theme(
            background_color="white",
            major_label_font_size=TICK_FONT_SIZE,
            minor_label_font_size=SUBTICK_FONT_SIZE,
            key_label_font_size=LEGEND_FONT_SIZE,
        ),
        # Guide.title("SQAVI vs MCMC pour xi"),
        Guide.xlabel("ξ"),
        Guide.ylabel("Densité"),
    );

    # x = 3*10^4:1:5*10^4;
    x = .01:.00001:.02;

    marginal = sqaviRes.approxMarginals[M+2];
    mcmcSample = mcmcChain[:, "κᵤ", 1].value[warmingSize:end];

    p4 = plot(
        layer(x=x, y=pdf.(marginal, x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSample, Geom.histogram(density=true, bincount=binCount)),
        Guide.Theme(
            background_color="white",
            major_label_font_size=TICK_FONT_SIZE,
            minor_label_font_size=SUBTICK_FONT_SIZE,
            key_label_font_size=LEGEND_FONT_SIZE,
        ),
        # Guide.title("SQAVI vs MCMC pour kappa_u"),
        Guide.xlabel("κᵤ"),
        Guide.ylabel("Densité"),
    );
    
    # x = 0:.1:300;
    x = 15:.001:23;

    marginal = sqaviRes.approxMarginals[M+3];
    mcmcSample = mcmcChain[:, "κᵥ", 1].value[warmingSize:end];

    p5 = plot(
        layer(x=x, y=pdf.(marginal, x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSample, Geom.histogram(density=true, bincount=binCount)),
        Guide.Theme(
            background_color="white",
            major_label_font_size=TICK_FONT_SIZE,
            minor_label_font_size=SUBTICK_FONT_SIZE,
            key_label_font_size=LEGEND_FONT_SIZE,
        ),
        # Guide.title("SQAVI vs MCMC pour kappa_v"),
        Guide.xlabel("κᵥ"),
        Guide.ylabel("Densité"),
    );

    # Create a legend plot
    legend_plot = plot(
        Guide.manual_color_key("", ["MCMC", "Approximation"], ["#56bcf9", "red"]),
        Guide.Theme(
            background_color="white",
            key_label_font_size=LEGEND_FONT_SIZE,
            key_position = :bottom,
        ),
        Coord.Cartesian(xmin=-0.5, xmax=0.5, ymin=-0.01, ymax=0.01),
    )

    p = vstack(legend_plot, p1, p2, p3, p4, p5)

    draw(SVG("$saveFolder/plots/sqavi_vs_mcmc_$numCell.svg"), p)
end


"""
    buildCAVImarginal(numCell, paramNum; sqaviRes)

Build the marginal density of a parameter of a given cell from the result of the CAVI algorithm.

# Arguments
TBD
"""
function buildCellCAVImarginal(numCell::Integer, paramNum::Integer; sqaviRes::SQAVIres)
    return Normal(
        params(sqaviRes.approxMarginals[numCell])[1][paramNum],
        sqrt(diag(params(sqaviRes.approxMarginals[numCell])[2])[paramNum])
    )
end

