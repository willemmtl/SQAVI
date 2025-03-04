using Mamba, Distances

"""
    compMCMCestimates(chain::Mamba.Chains, warmingSize::Integer)
"""
function compMCMCestimates(chain::Mamba.Chains, warmingSize::Integer)
    return mean(chain.value[warmingSize:end, :, 1], dims=1)[:]
end


"""
"""
function compDistance(chain::Mamba.Chains, gridTarget::Array{Float64, 3}, warmingSize::Integer)
    mcmcEstimates = compMCMCestimates(chain, warmingSize)[1:end-2];
    targetValues = vcat(
        gridTarget[:, :, 1]'[:],
        gridTarget[:, :, 2]'[:],
        [0.0],
    );
    return euclidean(mcmcEstimates, targetValues)
end


"""
"""
function compCAVIestimates(traces::Dict)
    return vcat(
        traces[:muMean][:, end],
        traces[:phiMean][:, end],
        traces[:xiMean][end],
    )
end


"""
"""
function compDistance(traces::Dict, gridTarget::Array{Float64, 3})
    caviEstimates = compCAVIestimates(traces);
    targetValues = vcat(
        gridTarget[:, :, 1]'[:],
        gridTarget[:, :, 2]'[:],
        [0.0],
    );
    return euclidean(caviEstimates, targetValues)
end