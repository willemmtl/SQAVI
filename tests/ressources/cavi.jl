using Extremes

struct Ressource
    traces::Dict
    caviCounter::Dict
    spatialScheme::Dict
end


"""
initialize(caviCounter, spatialScheme)
"""
M = 4;
μ = 40.0;
ϕ = 2.0;
ξ = 0.05;

initializeRessource = Ressource(
    Dict(),
    Dict(
        :iter => 0,
        :epoch => 0,
        :numCell => 1,
    ),
    Dict(
        :M => M,
        :Fmu => iGMRF(2, 2, 1, 10.0),
        :Fphi => iGMRF(2, 2, 1, 100.0),
        :data => [rand(GeneralizedExtremeValue(μ, exp(ϕ), ξ), 1000) for _ = 1:M]
    ),    
)


"""
compApproxMarginals!(approxMarginals, traces; caviCounter, spatialScheme)
"""

M = 4;

# identity matrix for all cells
cellVar_iter1 = zeros(4, M);
cellVar_iter1[1, :] = ones(M);
cellVar_iter1[4, :] = ones(M);
cellVar_iter2 = zeros(4, M);
cellVar_iter2[1, :] = ones(M);
cellVar_iter2[4, :] = ones(M);
cellVar = cat(cellVar_iter1, cellVar_iter2, dims=3);

cAM!ressource = Ressource(
    Dict(
        :muMean => zeros(M, 2),
        :phiMean => zeros(M, 2),
        :xiMean => zeros(2),
        :cellVar => cellVar,
        :kappaUparams => [1; 2;;],
        :kappaVparams => [1; 2;;],
    ),
    Dict(
        :iter => 1,
        :epoch => 1,
        :numCell => 1,
    ),
    Dict(
        :M => M,
        :Fmu => iGMRF(2, 2, 1, 10.0),
        :Fphi => iGMRF(2, 2, 1, 100.0),
        :data => [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
    ),
);


"""
updateParams!(traces, caviCounter, spatialScheme)
"""

nEpoch = 2;
epochSize = 1;
M = 4;

uP!ressource = Ressource(
    Dict(
        :muMean => zeros(M, nEpoch*epochSize),
        :phiMean => zeros(M, nEpoch*epochSize),
        :xiMean => zeros(nEpoch*epochSize),
        :cellVar => Matrix{Float64}(undef,(4, M)),
        :kappaUparams => [1.0 1.0; 2.0 1.0],
        :kappaVparams => [1.0 1.0; 2.0 1.0],
    ),
    Dict(
        :iter => 2,
        :epoch => 1,
        :numCell => 1,
    ),
    Dict(
        :M => M,
        :Fmu => iGMRF(2, 2, 1, 10.0),
        :Fphi => iGMRF(2, 2, 1, 100.0),
        :data => [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
    ),
);


"""
runIter!(traces; caviCounter, spatialScheme)
"""

M = 4;

runIter!ressource = Ressource(
    Dict(
        :muMean => zeros(M),
        :phiMean => zeros(M),
        :xiMean => [0.0],
        :cellVar => fill(NaN, 4, M),
        :kappaUparams => [1; 2;;],
        :kappaVparams => [1; 2;;],
    ),
    Dict(
        :iter => 0,
        :epoch => 1,
        :numCell => 1,
    ),
    Dict(
        :M => M,
        :Fmu => iGMRF(2, 2, 1, 10.0),
        :Fphi => iGMRF(2, 2, 1, 100.0),
        :data => [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
    ),
);

"""
runCAVI(nEpochMax, epochSize, initialValues, spatialScheme, ϵ)
"""

data = [
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0, 1.0],
];
M₂ = 2;
M₁ = 2;
M = M₁ * M₂;

runCAVIressource = Dict(
    :nEpochMax => 2,
    :epochSize => 2,
    :initialValues => Dict(
        :μ => zeros(M),
        :ϕ => zeros(M),
        :ξ => 0.0,
        :kappaUparam => ((M - 1) / 2 + 1) / 100,
        :kappaVparam => ((M - 1) / 2 + 1) / 100,
    ),
    :spatialScheme => Dict(
        :M => M,
        :Fmu => iGMRF(M₁, M₂, 1, 1),
        :Fphi => iGMRF(M₁, M₂, 1, 1),
        :data => data,
    ),
)