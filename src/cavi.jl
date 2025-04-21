include("utils.jl");
include("model.jl");
include("monteCarlo.jl");

struct CAVIres
    MCKL::DenseVector
    approxMarginals::Vector{<:Distribution}
    traces::Dict
end


"""
    runCAVI(nEpochMax, epochSize, initialValues, spatialScheme, ϵ)

CAVI algorithm.
"""
function runCAVI(
    nEpochMax::Integer,
    epochSize::Integer,
    initialValues::Dict{Symbol, Any},
    spatialScheme::Dict{Symbol, Any},
    ϵ::Real=0.05,
)
    
    M = spatialScheme[:M];

    approxMarginals = Vector{Distribution}(undef, M+3);

    MCKL = Vector{Float64}();

    traces = initialize(initialValues);

    caviCounter = Dict(
        :iter => 1,
        :epoch => 1,
        :numCell => 1,
    )

    println("Itération 0...")
    compApproxMarginals!(approxMarginals, traces, caviCounter=caviCounter, spatialScheme=spatialScheme);
    compMCKL!(MCKL, approxMarginals, caviCounter=caviCounter, spatialScheme=spatialScheme);

    while (caviCounter[:epoch] <= nEpochMax)
        caviCounter[:epoch] += 1;
        runEpoch!(traces, MCKL, approxMarginals, caviCounter=caviCounter, epochSize=epochSize, spatialScheme=spatialScheme);
        if (caviCounter[:epoch] >= 2)
            if (abs(MCKL[caviCounter[:epoch]] - MCKL[caviCounter[:epoch]-1]) < ϵ)
                println("L'algorithme a convergé !")
                return CAVIres(
                    MCKL,
                    approxMarginals,
                    traces,
                )
            end
        end
    end

    println("L'algorithme a atteint le nombre maximum d'itérations.")

    return CAVIres(
        MCKL,
        approxMarginals,
        traces,
    )
end


"""
    initialize(initialValues)

Set initial values.

# Arguments
- `initialValues::Dict`: Initial values to assign.
"""
function initialize(initialValues::Dict)

    # Identity Matrices
    cellVar = zeros(4, M);
    cellVar[1, :] = ones(M);
    cellVar[4, :] = ones(M);

    return Dict(
        :muMean => reshape(initialValues[:μ], M, 1),
        :phiMean => reshape(initialValues[:ϕ], M, 1),
        :xiMean => [initialValues[:ξ]],
        :cellVar => cellVar,
        :kappaUparams => [(M - 1)/2 + 1, initialValues[:kappaUparam]],
        :kappaVparams => [(M - 1)/2 + 1, initialValues[:kappaVparam]],
    )

end


"""
    runEpoch!(traces, MCKL, approxMarginals; caviCounter, epochSize, spatialScheme)

Perform one epoch of the CAVI algorithm.

# Arguments
- `traces::Dict`: Traces of all variational parameters.
- `MCKL::DenseVector`: Trace of KL divergence values.
- `approxMarginals::Vector{<:Distribution}`: Current approximation marginals.
- `caviCounter::Dict`: Counters of the CAVI algorithm.
- `epochSize::Integer`: Number of iterations within the epoch.
- `spatialScheme::Dict`: Spatial structures and data.
"""
function runEpoch!(traces::Dict, MCKL::DenseVector, approxMarginals::Vector{<:Distribution}; caviCounter::Dict, epochSize::Integer, spatialScheme::Dict)
    
    for _ = 1:epochSize
        runIter!(traces, caviCounter=caviCounter, spatialScheme=spatialScheme);
    end
    
    compApproxMarginals!(approxMarginals, traces, caviCounter=caviCounter, spatialScheme=spatialScheme);
    compMCKL!(MCKL, approxMarginals, caviCounter=caviCounter, spatialScheme=spatialScheme);
end


"""
    compMCKL!(MCKL, approxMarginals; caviCounter, spatialScheme)

Compute KL divergence between target and approximation densities using Monte Carlo.

# Arguments
- `MCKL::DenseVector`: Trace of KL divergence values.
- `approxMarginals::Vector{<:Distribution}`: Current approximation marginals.
- `caviCounter::Dict`: Counters of the CAVI algorithm.
- `spatialScheme::Dict`: Spatial structures and data.
"""
function compMCKL!(MCKL::DenseVector, approxMarginals::Vector{<:Distribution}; caviCounter::Dict, spatialScheme::Dict)

    Fmu = spatialScheme[:Fmu];
    Fphi = spatialScheme[:Fphi];
    data = spatialScheme[:data];

    logtarget(θ::DenseVector) = logposterior(θ, Fmu=Fmu, Fphi=Fphi, data=data);
    push!(MCKL, MonteCarloKL(logtarget, approxMarginals));

end


"""
    compApproxMarginals!(approxMarginals, traces; caviCounter, spatialScheme)

Build the approximation marginals thanks to the current values of variational parameters.

# Arguments
- `approxMarginals::Vector{<:Distribution}`: Current approximation marginals.
- `traces::Dict`: Traces of all variational parameters.
- `caviCounter::Dict`: Counters of the CAVI algorithm.
- `spatialScheme::Dict`: Spatial structures and data.
"""
function compApproxMarginals!(approxMarginals::Vector{<:Distribution}, traces::Dict; caviCounter::Dict, spatialScheme::Dict)

    iter = caviCounter[:iter];
    M = spatialScheme[:M];

    for i = 1:M

        m_i = [
            traces[:muMean][i, iter],
            traces[:phiMean][i, iter],
        ];
        cellVar = buildVar(traces[:cellVar][:, i, iter]);

        try
            approxMarginals[i] = MvNormal(m_i, round.(cellVar, digits = 12));
        catch exc
            if isa(exc, PosDefException)
                println("La matrice de variance de la cellule $i à l'itération $iter n'est pas définie positive !")
            end
        end
        
    end

    xiVar = fisherVar(ξ -> xilfc(ξ, caviCounter, traces, spatialScheme), traces[:xiMean][iter]);
    approxMarginals[M+1] = Normal(traces[:xiMean][iter], sqrt(xiVar));
    approxMarginals[M+2] = Gamma(traces[:kappaUparams][1, iter], 1/traces[:kappaUparams][2, iter]);
    approxMarginals[M+3] = Gamma(traces[:kappaVparams][1, iter], 1/traces[:kappaVparams][2, iter]);

end


"""
    runIter!(traces; caviCounter, spatialScheme)

Perform one iteration of the CAVI algorithm.

# Arguments
- `traces::Dict`: Traces of all variational parameters.
- `caviCounter::Dict`: Counters of the CAVI algorithm.
- `spatialScheme::Dict`: Spatial structures and data.
"""
function runIter!(traces::Dict; caviCounter::Dict, spatialScheme::Dict)

    caviCounter[:iter] += 1
    iter = caviCounter[:iter];
    M = spatialScheme[:M];

    println("Itération $(iter-1)...")

    traces[:muMean] = hcat(traces[:muMean], traces[:muMean][:, iter-1]);
    traces[:phiMean] = hcat(traces[:phiMean], traces[:phiMean][:, iter-1]);
    push!(traces[:xiMean], traces[:xiMean][iter-1]);
    traces[:kappaVparams] = hcat(traces[:kappaVparams], [(M - 1)/2 + 1, traces[:kappaVparams][2, iter-1]]);
    traces[:kappaUparams] = hcat(traces[:kappaUparams], [(M - 1)/2 + 1, traces[:kappaUparams][2, iter-1]]);

    updateParams!(traces, caviCounter, spatialScheme);

end


"""
    updateParams!(traces, caviCounter, spatialScheme)

Perform one iteration of the CAVI algorithm.

# Arguments
- `traces::Dict`: Traces of all variational parameters.
- `caviCounter::Dict`: Counters of the CAVI algorithm.
- `spatialScheme::Dict`: Spatial structures and data.
"""
function updateParams!(traces::Dict, caviCounter::Dict, spatialScheme::Dict)

    iter = caviCounter[:iter];
    M = spatialScheme[:M];
    cellsVar = Matrix{Float64}(undef, (4, M))

    for i = 1:M

        caviCounter[:numCell] = i;
        θ₀ = [
            traces[:muMean][i, iter],
            traces[:phiMean][i, iter],
        ];
        
        (m_i, cellVar) = compCellQuadraticApprox(θ₀, caviCounter, traces, spatialScheme);

        (traces[:muMean][i, iter], traces[:phiMean][i, iter]) =  m_i;
        cellsVar[:, i] = flatten(cellVar);
        
    end

    traces[:cellVar] = cat(traces[:cellVar], cellsVar, dims=3);

    traces[:xiMean][iter] = findMode(ξ -> xilfc(ξ, caviCounter, traces, spatialScheme), traces[:xiMean][iter])[1];
    traces[:kappaUparams][2, iter] = compKappaParam(traces[:muMean][:, iter], traces[:cellVar][1, :, iter], spatialScheme[:Fmu]);
    traces[:kappaVparams][2, iter] = compKappaParam(traces[:phiMean][:, iter], traces[:cellVar][4, :, iter], spatialScheme[:Fphi]);

end


"""
    compCellQuadraticApprox(θ₀, caviCounter, traces, spatialScheme)

Compute mean and variance of the Normal approximation of the cell's full conditional.

# Arguments :
- `θ₀::DenseVector`: Initial value to find the mode.
- `caviCounter::Dict`: Counters of the CAVI algorithm.
- `traces::Dict`: Traces of all variational parameters.
- `spatialScheme::Dict`: Spatial structures and data.
"""
function compCellQuadraticApprox(
    θ₀::DenseVector,
    caviCounter::Dict,
    traces::Dict,
    spatialScheme::Dict,
)

    mode = findMode(θi -> clfc(θi, caviCounter, traces, spatialScheme), θ₀);
    
    return mode, fisherVar(θi -> clfc(θi, caviCounter, traces, spatialScheme), mode);

end


"""
Log full conditional density of [μi, ϕi] knowing all other parameters.
"""
function clfc(θi::DenseVector, caviCounter::Dict, traces::Dict, spatialScheme::Dict)

    numCell = caviCounter[:numCell];
    iter = caviCounter[:iter];

    Fmu = spatialScheme[:Fmu];
    Fphi = spatialScheme[:Fphi];
    data = spatialScheme[:data];

    μ = traces[:muMean][:, iter];
    ϕ = traces[:phiMean][:, iter];
    ξ = traces[:xiMean][iter];
    
    μ̄i = neighborsMean(numCell, μ, Fmu);
    ϕ̄i = neighborsMean(numCell, ϕ, Fphi);
    κ̂ᵤ =  estimateKappa("u", traces=traces, caviCounter=caviCounter);
    κ̂ᵥ =  estimateKappa("v", traces=traces, caviCounter=caviCounter);

    return celllogfullconditional(
        numCell,
        θi,
        ξ=ξ,
        μ̄i=μ̄i,
        ϕ̄i=ϕ̄i,
        κ̂ᵤ=κ̂ᵤ,
        κ̂ᵥ=κ̂ᵥ,
        Fmu=Fmu,
        Fphi=Fphi,
        data=data,
    )

end


"""
Log full conditional density of ξ knowing all other variables.
"""
function xilfc(ξ::Real, caviCounter::Dict, traces::Dict, spatialScheme::Dict)
    
    iter = caviCounter[:iter];

    data = spatialScheme[:data];

    Eμ = traces[:muMean][:, iter];
    Eϕ = traces[:phiMean][:, iter];
    varϕ = traces[:cellVar][4, :, iter];

    return xilogfullconditional(
        ξ,
        Eμ=Eμ,
        Eϕ=Eϕ,
        varϕ=varϕ,
        data=data,
    )

end


"""
    estimateKappa(variant; traces, caviCounter)

Compute kappa estimate based on the parameters of its law (Gamma).

# Arguments
- `variant::String`: Which kappa to compute, 'u' or 'v'.
- `traces::Dict`: Traces of all variational parameters.
- `caviCounter::Dict`: Counters of the CAVI algorithm.
"""
function estimateKappa(variant::String; traces::Dict, caviCounter::Dict)
    
    iter = caviCounter[:iter];

    if (variant == "u")
        return traces[:kappaUparams][1, iter] / traces[:kappaUparams][2, iter];
    elseif (variant == "v")
        return traces[:kappaVparams][1, iter] / traces[:kappaVparams][2, iter];
    end
end


"""
    buildVar(components)

Build the variance matrix Σᵢ from the vector of values.

# Arguments
- `components::DenseVector`: Value of each component.
"""
function buildVar(components::DenseVector)
    return reshape(components, 2, 2)'
end