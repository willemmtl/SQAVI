struct Ressource
    traces::Dict
    caviCounter::Dict
    spatialScheme::Dict
end


"""
compApproxMarginals!(approxMarginals, traces; caviCounter, spatialScheme)
"""

nEpoch = 1;
epochSize = 1;
M = 4;

cellVar = fill(1.0, M, nEpoch*epochSize, 4);
cellVar[:, 1, 2] = zeros(M);
cellVar[:, 1, 3] = zeros(M);

cAM!ressource = Ressource(
    Dict(
        :muMean => zeros(M, nEpoch*epochSize),
        :phiMean => zeros(M, nEpoch*epochSize),
        :xiMean => zeros(nEpoch*epochSize),
        :cellVar => cellVar,
        :xiVar => ones(M),
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
        :cellVar => zeros(M, nEpoch*epochSize, 4),
        :xiVar => ones(M),
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