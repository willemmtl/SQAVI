using GMRF, ProgressMeter, LinearAlgebra, Mamba, Serialization, Extremes, StructArrays, Suppressor
using Distributions:loglikelihood

include("model.jl");

"""
    mcmc(datastructure, niter, stepsize; saveFolder)

Metropolis algorithm.
Use Markov asumption to parallelize calculations.

# Arguments
- `datastructure::Dict`: Data and spatial schemes.
- `niter::Integer`: Number of iterations.
- `stepsize::Dict`: Instrumental variance's step size for each parameter.
- `saveFolder::String`: Folder where to save the resulting chain. 
"""
function mcmc(datastructure::Dict, niter::Integer, stepsize::Dict; saveFolder::String)

    Y = datastructure[:Y];
    Fmu = datastructure[:Fmu];
    Fphi = datastructure[:Fphi];

    M = prod(Fmu.G.gridSize);

    μ = Array{Float64}(undef, M, niter);
    ϕ = Array{Float64}(undef, M, niter);
    ξ = Array{Float64}(undef, niter);
    κᵤ = Array{Float64}(undef, niter);
    κᵥ = Array{Float64}(undef, niter);

    # Initialisation
    
    fittedgev = @suppress begin
        StructArray(gevfit.(Y));
    end
    MLEs = hcat(fittedgev.θ̂...);

    μ[:, 1] = MLEs[1, :];
    ϕ[:, 1] = MLEs[2, :];
    ξ[1] = mean(MLEs[3, :]);
    κᵤ[1] = 1;
    κᵥ[1] = 1;

    Fmu = iGMRF(Fmu.G.gridSize..., Fmu.rankDeficiency, κᵤ[1]);
    Fphi = iGMRF(Fphi.G.gridSize..., Fphi.rankDeficiency, κᵥ[1]);

    @showprogress for j=2:niter
        
        μ[:, j] = μ[:, j-1];
        ϕ[:, j] = ϕ[:, j-1];
        ξ[j] = ξ[j-1];
        κᵤ[j] = κᵤ[j-1];
        κᵥ[j] = κᵥ[j-1];
        
        # Pour μ
        μ̃ = μ[:, j] .+ stepsize[:μ] .* randn(M);
        logL = datalevel_loglike.(Y, μ̃, ϕ[:, j], ξ[j]) - datalevel_loglike.(Y, μ[:, j], ϕ[:, j], ξ[j]);
        μ[:, j] = vertices_update(μ[:, j], Fmu, μ̃, logL);

        # Pour κᵤ
        κᵤ[j] = samplePrecision(Fmu, μ[:, j]);

        # Pour ϕ
        ϕ̃ = ϕ[:, j] .+ stepsize[:ϕ] .* randn(M);
        logL = datalevel_loglike.(Y, μ[:, j], ϕ̃, ξ[j]) - datalevel_loglike.(Y, μ[:, j], ϕ[:, j], ξ[j]);
        ϕ[:, j] = vertices_update(ϕ[:, j], Fphi, ϕ̃, logL);

        # Pour κᵥ
        κᵥ[j] = samplePrecision(Fphi, ϕ[:, j]);
        
        # Pour ξ
        ξ̃ = ξ[j] .+ stepsize[:ξ] .* randn();
        ξ[j] = xi_update(ξ[j], ξ̃, μ[:, j], ϕ[:, j], Y);

        # Mise à jour des iGMRF
        Fmu = iGMRF(Fmu.G.gridSize..., Fmu.rankDeficiency, κᵤ[j]);
        Fphi = iGMRF(Fphi.G.gridSize..., Fphi.rankDeficiency, κᵥ[j]);

    end

    chain = createChain(M, μ, ϕ, ξ, κᵤ, κᵥ)
    saveChain!(chain, saveFolder);
    return chain

end


"""
    saveChain!(chain, saveFolder)

Save the MCMC chains in the given saveFolder.
"""
function saveChain!(chain::Mamba.Chains, saveFolder::String)
    open("$saveFolder/mcmc_chain.dat", "w") do file
        serialize(file, chain)
    end
end


"""
    loadChain(folderName)

Load a CAVI result previously saved in the given folderName.
The file must have the name 'cavires.dat' which is automatically given by saveRes!(res, saveFolder) function.
"""
function loadChain(folderName::String)
    return open("$folderName/mcmc_chain.dat", "r") do file
        deserialize(file)
    end
end


"""
    xi_update(ξ, ξ̃, μ, ϕ, Y)

Choose whether to accept xi's candidate or not based on the Metropolis criterion.
"""
function xi_update(ξ::Real, ξ̃::Real, μ::DenseVector, ϕ::DenseVector, Y::Vector{Vector{T}}) where T<:AbstractFloat

    lr = logfxi(ξ̃, μ, ϕ, Y) - logfxi(ξ, μ, ϕ, Y);

    if lr > log(rand())
        return ξ̃
    else
        return ξ
    end
end


"""
    logfxi(ξ, μ, ϕ, Y)

Xi's log full conditional.
"""
function logfxi(ξ::Real, μ::DenseVector, ϕ::DenseVector, Y::Vector{Vector{T}}) where T<:AbstractFloat
    return sum(datalevel_loglike.(Y, μ, ϕ, ξ)) + logpdf(Beta(6, 9), ξ + .5)
end



"""
    datalevel_loglike(Y, μ, ϕ, ξ)

Compute the likelihood of the GEV parameters μ, ϕ and ξ as function of the data Y.

# Arguments
- `Y` : Vector of data.
- `μ` : GEV location parameter.
- `ϕ` : GEV log-scale parameter.
- `ξ` : GEV shape parameter.
"""
function datalevel_loglike(Y::DenseVector, μ::Real, ϕ::Real, ξ::Real)
    return loglikelihood(GeneralizedExtremeValue(μ, exp(ϕ), ξ), Y)
end


"""
    vertices_update(θ, F, θ̃, logL)

Update the vertices using the proposed candidates with data.

# Arguments
- `θ::DenseVector`: The current state.
- `F::iGMRF`: Spatial scheme.
- `θ̃::DenseVector`: Proposed candidates.
- `logL::DenseVector`: The log-likelihood difference between the proposed and the current states at every vertice.
"""
function vertices_update(θ::DenseVector, F::iGMRF, θ̃::DenseVector, logL::DenseVector)

    for indices in F.G.condIndSubset
        vertices_update!(θ, F, θ̃, logL, indices);
    end

    return θ

end


"""
    vertices_update!(θ, F, θ̃, logL, ind)

Update the vertices within a conditional independant subset using the proposed candidates with data.
The vertices are assumed to be conditionally independant, so they can be updated in parallel.

# Arguments
- `θ::DenseVector`: The current state.
- `F::iGMRF`: Spatial scheme.
- `θ̃::DenseVector`: Proposed candidates.
- `logL::DenseVector`: The log-likelihood difference between the proposed and the current states at every vertice.
- `ind::Vector{Integer}`: Indices of the current conditional independant subset.
"""
function vertices_update!(θ::DenseVector, F::iGMRF, θ̃::DenseVector, logL::DenseVector, ind::Vector{Int64})
    
    pds = fullconditionalsIGMRF(F, θ)[ind];
    lf = logpdf.(pds, θ̃[ind]) .- logpdf.(pds, θ[ind]);
    lr = logL[ind] .+ lf;

    logu = log.(rand(length(ind)));

    accepted = lr .> logu;
    
    setindex!(θ, θ̃[ind][accepted], ind[accepted]);

end


"""
    fullconditionalsIGMRF(F, θ)

Compute the probability density of the full conditional function of the GEV's location parameter due to the iGMRF.

# Arguments

- `F::iGMRF`: Spatial scheme.
- `θ::Vector{<:Real}`: Last updated parameters.
"""
function fullconditionalsIGMRF(F::iGMRF, θ::Vector{<:Real})

    Q = F.κ * Array(diag(F.G.W))
    b = -F.κ * (F.G.W̄ * θ)

    return NormalCanon.(b, Q)

end


function samplePrecision(F::iGMRF, θ::DenseVector)
    
    M = prod(F.G.gridSize);
    r = F.rankDeficiency;

    α = (M - r) / 2 + 1;
    β = θ' * F.G.W * θ / 2 + 0.01;

    return rand(Gamma(α, 1/β))
end


"""
    createChain(M, μ, ϕ, ξ, κᵤ, κᵥ)

Create an object of type Mamba.Chains from the output of the mcmc algorithm.

# Arguments
- `M::Integer`: Number of cells.
- `μ::Matrix{<:Real}`: Traces of location parameters.
- `ϕ::Matrix{<:Real}`: Traces of log-scale parameters.
- `ξ::Vector{<:Real}`: Trace of shape parameters.
- `κᵤ::Vector{<:Real}`: Trace of precision parameter of Fmu.
- `κᵥ::Vector{<:Real}`: Trace of precision parameter of Fphi.
"""
function createChain(M::Integer, μ::Matrix{<:Real}, ϕ::Matrix{<:Real}, ξ::Vector{<:Real}, κᵤ::Vector{<:Real}, κᵥ::Vector{<:Real})
    
    paramnames = vcat(["μ$i" for i=1:M], ["ϕ$i" for i=1:M], ["ξ"], ["κᵤ"], ["κᵥ"]);
    res = vcat(μ, ϕ, ξ', κᵤ', κᵥ');
    
    return Mamba.Chains(collect(res'), names=paramnames)
end