using Serialization

include("utils.jl");
include("model.jl");
include("monteCarlo.jl");

struct CAVIres
    MCKL::DenseVector
    approxMarginals::Vector{<:Distribution}
    traces::Dict
end


"""
    runCAVI(nEpochMax, epochSize, initialValues, spatialScheme, ϵ; saveFolder)

CAVI algorithm.
"""
function runCAVI(
    nEpochMax::Integer,
    epochSize::Integer,
    spatialScheme::Dict{Symbol, Any},
    ϵ::Real=0.05;
    initialValues::Union{Dict{Symbol, Any}, String},
    saveFolder::String,
)
    
    caviCounter = Dict(
        :iter => 0,
        :epoch => 0,
        :numCell => 1,
    )

    if isa(initialValues, Dict)
        (traces, approxMarginals, MCKL) = initialize(initialValues, caviCounter, spatialScheme, saveFolder=saveFolder);
    elseif isa(initialValues, String)
        (traces, approxMarginals, MCKL) = initialize(initialValues);
    end

    while (caviCounter[:epoch] < nEpochMax)
        runEpoch!(traces, MCKL, approxMarginals, caviCounter=caviCounter, epochSize=epochSize, spatialScheme=spatialScheme, saveFolder=saveFolder);
        if (caviCounter[:epoch] >= 2)
            if (abs(MCKL[caviCounter[:epoch]] - MCKL[caviCounter[:epoch]-1]) < ϵ)
                println("L'algorithme a convergé !")
                res = CAVIres(MCKL, approxMarginals, traces);
                saveRes!(res, saveFolder);
                return res
            end
        end
    end

    println("L'algorithme a atteint le nombre maximum d'itérations.")

    res = CAVIres(MCKL, approxMarginals, traces);
    saveRes!(res, saveFolder);

    return res
end


"""
    saveRes!(res, saveFolder)

Save the whole CAVI results in the given saveFolder.
"""
function saveRes!(res::CAVIres, saveFolder::String)
    open("$saveFolder/cavires.dat", "w") do file
        serialize(file, res)
    end
end


"""
    loadRes(folderName)

Load a CAVI result previously saved in the given folderName.
The file must have the name 'cavires.dat' which is automatically given by saveRes!(res, saveFolder) function.
"""
function loadRes(folderName::String)
    return open("$folderName/cavires.dat", "r") do file
        deserialize(file)
    end
end


"""
    initialize(initialValues, caviCounter, spatialScheme)

Set initial values.
Compute approxMarginals and MCKL with initialize values.

# Arguments
- `initialValues::Dict{Symbol, Any}`: Initial values to assign.
- `caviCounter::Dict`: Counters of the CAVI algorithm.
- `spatialScheme::Dict`: Spatial structures and data.
- `saveFolder::String` : Folder where to save current results.
"""
function initialize(
    initialValues::Dict{Symbol, Any},
    caviCounter::Dict,
    spatialScheme::Dict;
    saveFolder::String,
)

    println("Itération 0...")
    
    M = spatialScheme[:M];
    approxMarginals = Vector{Distribution}(undef, M+3);
    MCKL = Vector{Float64}();

    # Identity Matrices
    cellVar = zeros(4, M);
    cellVar[1, :] = ones(M);
    cellVar[4, :] = ones(M);

    traces = Dict(
        :muMean => reshape(initialValues[:μ], M, 1),
        :phiMean => reshape(initialValues[:ϕ], M, 1),
        :xiMean => [initialValues[:ξ]],
        :cellVar => cellVar,
        :kappaUparams => [(M - 1)/2 + 1, initialValues[:kappaUparam]],
        :kappaVparams => [(M - 1)/2 + 1, initialValues[:kappaVparam]],
    )
    compApproxMarginals!(approxMarginals, traces, caviCounter=caviCounter, spatialScheme=spatialScheme);
    saveApproxMarginals!(approxMarginals, saveFolder);
    compMCKL!(MCKL, approxMarginals, spatialScheme=spatialScheme);
    saveMCKL!(MCKL, saveFolder);

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
    runEpoch!(traces, MCKL, approxMarginals; caviCounter, epochSize, spatialScheme, saveFolder)

Perform one epoch of the CAVI algorithm.

# Arguments
- `traces::Dict`: Traces of all variational parameters.
- `MCKL::DenseVector`: Trace of KL divergence values.
- `approxMarginals::Vector{<:Distribution}`: Current approximation marginals.
- `caviCounter::Dict`: Counters of the CAVI algorithm.
- `epochSize::Integer`: Number of iterations within the epoch.
- `spatialScheme::Dict`: Spatial structures and data.
- `saveFolder::String` : Folder where to save current results.
"""
function runEpoch!(traces::Dict, MCKL::DenseVector, approxMarginals::Vector{<:Distribution}; caviCounter::Dict, epochSize::Integer, spatialScheme::Dict, saveFolder::String)
    
    caviCounter[:epoch] += 1;

    for _ = 1:epochSize
        runIter!(traces, caviCounter=caviCounter, spatialScheme=spatialScheme, saveFolder=saveFolder);
    end
    
    compApproxMarginals!(approxMarginals, traces, caviCounter=caviCounter, spatialScheme=spatialScheme);
    saveApproxMarginals!(approxMarginals, saveFolder);
    compMCKL!(MCKL, approxMarginals, spatialScheme=spatialScheme);
    saveMCKL!(MCKL, saveFolder);
end


"""
    saveApproxMarginals!(approxMarginals, saveFolder)

Save the current approximation marginals at the given saveFolder.
"""
function saveApproxMarginals!(approxMarginals::Vector{<:Distribution}, saveFolder::String)
    open("$saveFolder/approxMarginals.dat", "w") do file
        serialize(file, approxMarginals)
    end
end


"""
    saveMCKL!(MCKL, saveFolder)

Save the current MCKL vector at the given saveFolder.
"""
function saveMCKL!(MCKL::DenseVector, saveFolder::String)
    open("$saveFolder/MCKL.dat", "w") do file
        serialize(file, MCKL)
    end
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
    approxMarginals[M+1] = Normal(traces[:xiMean][end], sqrt(xiVar));
    approxMarginals[M+2] = Gamma(traces[:kappaUparams][1, end], 1/traces[:kappaUparams][2, end]);
    approxMarginals[M+3] = Gamma(traces[:kappaVparams][1, end], 1/traces[:kappaVparams][2, end]);

end


"""
    runIter!(traces; caviCounter, spatialScheme, saveFolder)

Perform one iteration of the CAVI algorithm.

# Arguments
- `traces::Dict`: Traces of all variational parameters.
- `caviCounter::Dict`: Counters of the CAVI algorithm.
- `spatialScheme::Dict`: Spatial structures and data.
- `saveFolder::String` : Folder where to save current results.
"""
function runIter!(traces::Dict; caviCounter::Dict, spatialScheme::Dict, saveFolder::String)

    caviCounter[:iter] = length(traces[:xiMean]);
    iter = caviCounter[:iter];
    M = spatialScheme[:M];

    println("Itération $iter...")

    traces[:muMean] = hcat(traces[:muMean], traces[:muMean][:, iter]);
    traces[:phiMean] = hcat(traces[:phiMean], traces[:phiMean][:, iter]);
    push!(traces[:xiMean], traces[:xiMean][iter]);
    traces[:kappaVparams] = hcat(traces[:kappaVparams], [(M - 1)/2 + 1, traces[:kappaVparams][2, iter]]);
    traces[:kappaUparams] = hcat(traces[:kappaUparams], [(M - 1)/2 + 1, traces[:kappaUparams][2, iter]]);

    updateParams!(traces, caviCounter, spatialScheme);

    saveTraces!(traces, saveFolder);
end


"""
    saveTraces!(traces, saveFolder)

Save the current traces at the given saveFolder.
"""
function saveTraces!(traces::Dict, saveFolder::String)
    open("$saveFolder/traces.dat", "w") do file
        serialize(file, traces)
    end
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

    M = spatialScheme[:M];
    cellsVar = Matrix{Float64}(undef, (4, M))

    for i = 1:M

        caviCounter[:numCell] = i;
        θ₀ = [
            traces[:muMean][i, end],
            traces[:phiMean][i, end],
        ];
        
        (m_i, cellVar) = compCellQuadraticApprox(θ₀, caviCounter, traces, spatialScheme);

        (traces[:muMean][i, end], traces[:phiMean][i, end]) =  m_i;
        cellsVar[:, i] = flatten(cellVar);
        
    end

    traces[:cellVar] = cat(traces[:cellVar], cellsVar, dims=3);
    
    traces[:xiMean][end] = findMode(ξ -> xilfc(ξ, traces, spatialScheme), traces[:xiMean][end])[1];
    traces[:kappaUparams][2, end] = compKappaParam(traces[:muMean][:, end], traces[:cellVar][1, :, end], spatialScheme[:Fmu]);
    traces[:kappaVparams][2, end] = compKappaParam(traces[:phiMean][:, end], traces[:cellVar][4, :, end], spatialScheme[:Fphi]);

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