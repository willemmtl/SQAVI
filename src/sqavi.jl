using Serialization, Extremes, StructArrays, Suppressor

include("utils.jl");
include("model.jl");
include("monteCarlo.jl");

struct SQAVIres
    MCKL::DenseVector
    approxMarginals::Vector{<:Distribution}
    traces::Dict
end


"""
    runSQAVI(nEpochMax, epochSize, initialValues, spatialScheme, ϵ; saveFolder)

SQAVI algorithm.

# Arguments:
- `nEpochMax::Integer`: Max number of epochs to run is convergence is not reached.
- `epochSize::Integer`: Number of iterations per epoch.
- `spatialScheme::Dict{Symbol, Any}`: Data and dependance structure of the model.
- `ϵ::Real`: Convergence criterion.
- `initialValues::Union{Dict{Symbol, Any}, String, Nothing}`: Initial values. If not specified, MLE values are used, as described in the memoire.
- `saveFolder::String`: Folder where to save the output.
"""
function runSQAVI(
    nEpochMax::Integer,
    epochSize::Integer,
    spatialScheme::Dict{Symbol, Any},
    ϵ::Real=0.001;
    initialValues::Union{Dict{Symbol, Any}, String, Nothing}=nothing,
    saveFolder::String,
)
    
    m = spatialScheme[:m];

    sqaviCounter = Dict(
        :iter => 0,
        :epoch => 0,
        :numCell => 1,
    )

    if isa(initialValues, Dict)
        (traces, approxMarginals, MCKL) = initialize(initialValues, sqaviCounter, spatialScheme);
    elseif isa(initialValues, String)
        (traces, approxMarginals, MCKL) = initialize(initialValues);
    elseif isnothing(initialValues)
        (traces, approxMarginals, MCKL) = initialize(sqaviCounter, spatialScheme);
    end

    while (sqaviCounter[:epoch] < nEpochMax)
        runEpoch!(traces, MCKL, approxMarginals, sqaviCounter=sqaviCounter, epochSize=epochSize, spatialScheme=spatialScheme, saveFolder=saveFolder);
        if (abs(MCKL[sqaviCounter[:epoch]+1] - MCKL[sqaviCounter[:epoch]]) / m < ϵ)
            println("L'algorithme a convergé !")
            res = SQAVIres(MCKL, approxMarginals, traces);
            saveRes!(res, saveFolder);
            return res
        end
    end

    println("L'algorithme a atteint le nombre maximum d'itérations.")

    res = SQAVIres(MCKL, approxMarginals, traces);
    saveRes!(res, saveFolder);

    return res
end


"""
    saveRes!(res, saveFolder)

Save the whole SQAVI results in the given saveFolder.
"""
function saveRes!(res::SQAVIres, saveFolder::String)
    open("$saveFolder/sqavires.dat", "w") do file
        serialize(file, res)
    end
end


"""
    loadRes(folderName)

Load a SQAVI result previously saved in the given folderName.
The file must have the name 'sqavires.dat' which is automatically given by saveRes!(res, saveFolder) function.
"""
function loadRes(folderName::String)
    return open("$folderName/sqavires.dat", "r") do file
        deserialize(file)
    end
end


"""
    initialize(initialValues, sqaviCounter, spatialScheme)

Set initial values.
Compute approxMarginals and MCKL with initialize values.

# Arguments
- `initialValues::Dict{Symbol, Any}`: Initial values to assign.
- `sqaviCounter::Dict`: Counters of the SQAVI algorithm.
- `spatialScheme::Dict`: Spatial structures and data.
"""
function initialize(
    initialValues::Dict{Symbol, Any},
    sqaviCounter::Dict,
    spatialScheme::Dict
)

    println("Itération 0...")
    
    m = spatialScheme[:m];
    approxMarginals = Vector{Distribution}(undef, m+3);
    MCKL = Vector{Float64}();

    # Identity Matrices
    cellVar = zeros(4, m);
    cellVar[1, :] = ones(m);
    cellVar[4, :] = ones(m);

    traces = Dict(
        :muMean => reshape(initialValues[:μ], m, 1),
        :phiMean => reshape(initialValues[:ϕ], m, 1),
        :xiMean => [initialValues[:ξ]],
        :cellVar => cellVar,
        :kappaUparams => [(m - 1)/2 + 1, initialValues[:kappaUparam]],
        :kappaVparams => [(m - 1)/2 + 1, initialValues[:kappaVparam]],
    )

    compApproxMarginals!(approxMarginals, traces, sqaviCounter=sqaviCounter, spatialScheme=spatialScheme);
    compMCKL!(MCKL, approxMarginals, spatialScheme=spatialScheme);

    return traces, approxMarginals, MCKL
    
end


"""
    initialize(initialValues)

Load previous traces to use as initialize values.

# Arguments
- `initialValues::string`: Folder name where the traces are stored.
    The traces file must have the name 'traces.dat'.
"""
function initialize(initialValues::String)
    
    res = loadRes(initialValues);
    iter = length(res.traces[:xiMean]); # Include iteration 0.
    println("Loaded traces from $iter previous itérations.");
    return res.traces, res.approxMarginals, res.MCKL

end


"""
    initialize(sqaviCounter, spatialScheme)

Initialize values with MLE of GEV params.

# Arguments
- `sqaviCounter::Dict`: Counters of the SQAVI algorithm.
- `spatialScheme::Dict`: Spatial structures and data.
"""
function initialize(sqaviCounter::Dict, spatialScheme::Dict)
    
    println("Itération 0...")
    
    m = spatialScheme[:m];
    data = spatialScheme[:data];

    approxMarginals = Vector{Distribution}(undef, m+3);
    MCKL = Vector{Float64}();

    # Identity Matrices
    cellVar = zeros(4, m);
    cellVar[1, :] = ones(m);
    cellVar[4, :] = ones(m);

    fittedgev = @suppress begin
        StructArray(gevfit.(data));
    end
    MLEs = hcat(fittedgev.θ̂...);

    traces = Dict(
        :muMean => reshape(MLEs[1, :], m, 1),
        :phiMean => reshape(MLEs[2, :], m, 1),
        :xiMean => [mean(MLEs[3, :])],
        :cellVar => cellVar,
        :kappaUparams => [(m - 1)/2 + 1, (m - 1)/2 + 1],
        :kappaVparams => [(m - 1)/2 + 1, (m - 1)/2 + 1],
    )

    compApproxMarginals!(approxMarginals, traces, sqaviCounter=sqaviCounter, spatialScheme=spatialScheme);
    compMCKL!(MCKL, approxMarginals, spatialScheme=spatialScheme);

    return traces, approxMarginals, MCKL

end


"""
    runEpoch!(traces, MCKL, approxMarginals; sqaviCounter, epochSize, spatialScheme, saveFolder)

Perform one epoch of the SQAVI algorithm.

# Arguments
- `traces::Dict`: Traces of all variational parameters.
- `MCKL::DenseVector`: Trace of KL divergence values.
- `approxMarginals::Vector{<:Distribution}`: Current approximation marginals.
- `sqaviCounter::Dict`: Counters of the SQAVI algorithm.
- `epochSize::Integer`: Number of iterations within the epoch.
- `spatialScheme::Dict`: Spatial structures and data.
- `saveFolder::String` : Folder where to save current results.
"""
function runEpoch!(traces::Dict, MCKL::DenseVector, approxMarginals::Vector{<:Distribution}; sqaviCounter::Dict, epochSize::Integer, spatialScheme::Dict, saveFolder::String)
    
    sqaviCounter[:epoch] += 1;

    for _ = 1:epochSize
        runIter!(traces, sqaviCounter=sqaviCounter, spatialScheme=spatialScheme, saveFolder=saveFolder);
        res = SQAVIres(MCKL, approxMarginals, traces);
        saveRes!(res, saveFolder);
    end
    
    compApproxMarginals!(approxMarginals, traces, sqaviCounter=sqaviCounter, spatialScheme=spatialScheme);
    compMCKL!(MCKL, approxMarginals, spatialScheme=spatialScheme);
end


"""
    compMCKL!(MCKL, approxMarginals; spatialScheme)

Compute KL divergence between target and approximation densities using Monte Carlo.

# Arguments
- `MCKL::DenseVector`: Trace of KL divergence values.
- `approxMarginals::Vector{<:Distribution}`: Current approximation marginals.
- `spatialScheme::Dict`: Spatial structures and data.
"""
function compMCKL!(MCKL::DenseVector, approxMarginals::Vector{<:Distribution}; spatialScheme::Dict)

    Fmu = spatialScheme[:Fmu];
    Fphi = spatialScheme[:Fphi];
    data = spatialScheme[:data];

    logtarget(θ::DenseVector) = logposterior(θ, Fmu=Fmu, Fphi=Fphi, data=data);
    push!(MCKL, MonteCarloKL(logtarget, approxMarginals));

end


"""
    compApproxMarginals!(approxMarginals, traces; sqaviCounter, spatialScheme)

Build the approximation marginals thanks to the current values of variational parameters.

# Arguments
- `approxMarginals::Vector{<:Distribution}`: Current approximation marginals.
- `traces::Dict`: Traces of all variational parameters.
- `sqaviCounter::Dict`: Counters of the SQAVI algorithm.
- `spatialScheme::Dict`: Spatial structures and data.
"""
function compApproxMarginals!(approxMarginals::Vector{<:Distribution}, traces::Dict; sqaviCounter::Dict, spatialScheme::Dict)

    iter = sqaviCounter[:iter];
    m = spatialScheme[:m];

    for i = 1:m

        m_i = [
            traces[:muMean][i, end],
            traces[:phiMean][i, end],
        ];
        cellVar = buildVar(traces[:cellVar][:, i, end]);

        try
            approxMarginals[i] = MvNormal(m_i, Symmetric(cellVar));
        catch exc
            if isa(exc, PosDefException)
                println("La matrice de variance de la cellule $i à l'itération $iter n'est pas définie positive !")
            end
        end
        
    end

    xiVar = fisherVar(ξ -> xilfc(ξ, traces, spatialScheme), traces[:xiMean][end]);
    approxMarginals[m+1] = Normal(traces[:xiMean][end], sqrt(xiVar));
    approxMarginals[m+2] = Gamma(traces[:kappaUparams][1, end], 1/traces[:kappaUparams][2, end]);
    approxMarginals[m+3] = Gamma(traces[:kappaVparams][1, end], 1/traces[:kappaVparams][2, end]);

end


"""
    runIter!(traces; sqaviCounter, spatialScheme, saveFolder)

Perform one iteration of the SQAVI algorithm.

# Arguments
- `traces::Dict`: Traces of all variational parameters.
- `sqaviCounter::Dict`: Counters of the SQAVI algorithm.
- `spatialScheme::Dict`: Spatial structures and data.
- `saveFolder::String` : Folder where to save current results.
"""
function runIter!(traces::Dict; sqaviCounter::Dict, spatialScheme::Dict, saveFolder::String)

    sqaviCounter[:iter] = length(traces[:xiMean]);
    iter = sqaviCounter[:iter];
    m = spatialScheme[:m];

    println("Itération $iter...")

    traces[:muMean] = hcat(traces[:muMean], traces[:muMean][:, iter]);
    traces[:phiMean] = hcat(traces[:phiMean], traces[:phiMean][:, iter]);
    push!(traces[:xiMean], traces[:xiMean][iter]);
    traces[:kappaVparams] = hcat(traces[:kappaVparams], [(m - 1)/2 + 1, traces[:kappaVparams][2, iter]]);
    traces[:kappaUparams] = hcat(traces[:kappaUparams], [(m - 1)/2 + 1, traces[:kappaUparams][2, iter]]);

    updateParams!(traces, sqaviCounter, spatialScheme);

end


"""
    updateParams!(traces, sqaviCounter, spatialScheme)

Perform one iteration of the SQAVI algorithm.

# Arguments
- `traces::Dict`: Traces of all variational parameters.
- `sqaviCounter::Dict`: Counters of the SQAVI algorithm.
- `spatialScheme::Dict`: Spatial structures and data.
"""
function updateParams!(traces::Dict, sqaviCounter::Dict, spatialScheme::Dict)

    m = spatialScheme[:m];
    cellsVar = Matrix{Float64}(undef, (4, m))

    for i = 1:m

        sqaviCounter[:numCell] = i;
        θ₀ = [
            traces[:muMean][i, end],
            traces[:phiMean][i, end],
        ];
        
        (m_i, cellVar) = compCellQuadraticApprox(θ₀, sqaviCounter, traces, spatialScheme);

        (traces[:muMean][i, end], traces[:phiMean][i, end]) =  m_i;
        cellsVar[:, i] = flatten(cellVar);
        
    end

    traces[:cellVar] = cat(traces[:cellVar], cellsVar, dims=3);
    
    traces[:xiMean][end] = findMode(ξ -> xilfc(ξ, traces, spatialScheme), traces[:xiMean][end])[1];
    traces[:kappaUparams][2, end] = compKappaParam(traces[:muMean][:, end], traces[:cellVar][1, :, end], spatialScheme[:Fmu]);
    traces[:kappaVparams][2, end] = compKappaParam(traces[:phiMean][:, end], traces[:cellVar][4, :, end], spatialScheme[:Fphi]);

end


"""
    compCellQuadraticApprox(θ₀, sqaviCounter, traces, spatialScheme)

Compute mean and variance of the Normal approximation of the cell's full conditional.

# Arguments :
- `θ₀::DenseVector`: Initial value to find the mode.
- `sqaviCounter::Dict`: Counters of the SQAVI algorithm.
- `traces::Dict`: Traces of all variational parameters.
- `spatialScheme::Dict`: Spatial structures and data.
"""
function compCellQuadraticApprox(
    θ₀::DenseVector,
    sqaviCounter::Dict,
    traces::Dict,
    spatialScheme::Dict,
)

    mode = findMode(θi -> clfc(θi, sqaviCounter, traces, spatialScheme), θ₀);
    
    return mode, fisherVar(θi -> clfc(θi, sqaviCounter, traces, spatialScheme), mode);

end


"""
Log full conditional density of [μi, ϕi] knowing all other parameters.
"""
function clfc(θi::DenseVector, sqaviCounter::Dict, traces::Dict, spatialScheme::Dict)

    numCell = sqaviCounter[:numCell];

    Fmu = spatialScheme[:Fmu];
    Fphi = spatialScheme[:Fphi];
    data = spatialScheme[:data];

    μ = traces[:muMean][:, end];
    ϕ = traces[:phiMean][:, end];
    ξ = traces[:xiMean][end];
    
    μ̄i = neighborsMean(numCell, μ, Fmu);
    ϕ̄i = neighborsMean(numCell, ϕ, Fphi);
    κ̂ᵤ =  estimateKappa("u", traces=traces);
    κ̂ᵥ =  estimateKappa("v", traces=traces);

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
function xilfc(ξ::Real, traces::Dict, spatialScheme::Dict)
    
    data = spatialScheme[:data];

    Eμ = traces[:muMean][:, end];
    Eϕ = traces[:phiMean][:, end];
    varϕ = traces[:cellVar][4, :, end];

    return xilogfullconditional(
        ξ,
        Eμ=Eμ,
        Eϕ=Eϕ,
        varϕ=varϕ,
        data=data,
    )

end


"""
    estimateKappa(variant; traces)

Compute kappa estimate based on the parameters of its law (Gamma).

# Arguments
- `variant::String`: Which kappa to compute, 'u' or 'v'.
- `traces::Dict`: Traces of all variational parameters.
"""
function estimateKappa(variant::String; traces::Dict)
    
    if (variant == "u")
        return traces[:kappaUparams][1, end] / traces[:kappaUparams][2, end];
    elseif (variant == "v")
        return traces[:kappaVparams][1, end] / traces[:kappaVparams][2, end];
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