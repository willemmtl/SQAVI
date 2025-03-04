using Distributions, GMRF
using Distributions:loglikelihood

"""
    logposterior(θ; Fmu, Fphi, data)
"""
function logposterior(θ::DenseVector; Fmu::iGMRF, Fphi::iGMRF, data::Vector{Vector{Float64}})
    
    M = prod(Fmu.G.gridSize);
    μ = θ[1:M];
    ϕ = θ[M+1:2*M];
    ξ = θ[2*M+1];
    κᵤ = θ[2*M+2];
    κᵥ = θ[2*M+3];

    return (
        sum(loglikelihood.(GeneralizedExtremeValue.(μ, exp.(ϕ), ξ), data))
        + (prod(Fmu.G.gridSize) - Fmu.rankDeficiency)/2 * log(κᵤ) - κᵤ/2 * μ' * Fmu.G.W * μ
        + (prod(Fphi.G.gridSize) - Fphi.rankDeficiency)/2 * log(κᵥ) - κᵥ/2 * ϕ' * Fphi.G.W * ϕ
        + logpdf(Gamma(1, 100), κᵤ)
        + logpdf(Gamma(1, 100), κᵥ)
        + logpdf(Beta(6, 9), ξ + .5)
    )
end


"""
    celllogfullconditional(i, θi; ξ, μ̄i, ϕ̄i, Fmu, Fphi, data)

Log full conditional density of [μi, ϕi] knowing all other parameters.

# Arguments
- `i::Integer`: Numero of the cell.
- `θi::Vector{Float64}`: parameters for cell i -> variables [μi, ϕi].
- `ξ::Real`: Last updated shape parameter.
- `μ̄i::Real`: Neighbors influence for location (iGMRF).
- `ϕ̄i::Real`: Neighbors influence for log-scale (iGMRF).
- `κ̂ᵤ::Real`: κᵤ estimate.
- `κ̂ᵥ::Real`: κᵥ estimate.
- `Fmu::iGMRF`: Spatial scheme for location.
- `Fphi::iGMRF`: Spatial scheme for log-scale.
- `data::Vector{Float64}`: Observations for every cells.
"""
function celllogfullconditional(
    i::Integer,
    θi::DenseVector;
    ξ::Real,
    μ̄i::Real,
    ϕ̄i::Real,
    κ̂ᵤ::Real,
    κ̂ᵥ::Real,
    Fmu::iGMRF,
    Fphi::iGMRF,
    data::Vector{Vector{Float64}},
)

    return (
        loglikelihood(GeneralizedExtremeValue(θi[1], exp(θi[2]), ξ), data[i])
        + logpdf(Normal(μ̄i, sqrt(1/Fmu.G.W[i, i]/κ̂ᵤ)), θi[1])
        + logpdf(Normal(ϕ̄i, sqrt(1/Fphi.G.W[i, i]/κ̂ᵥ)), θi[2])
    )
end


"""
    xilogfullconditional(ξ; μ, ϕ, data)

Compute the log full conditional of ξ parameter of cell cellIndex.

# Arguments
- `ξ::DenseVector`: Variable.
- `μ::DenseVector`: Value of μ at this cell.
- `ϕ::DenseVector`: Value of ϕ at this cell.
- `data::Vector{Vector{Float64}}`: Observations.
"""
function xilogfullconditional(
    ξ::Real;
    μ::DenseVector,
    ϕ::DenseVector,
    data::Vector{Vector{Float64}},
)
    return (
        sum(loglikelihood.(GeneralizedExtremeValue.(μ, exp.(ϕ), ξ), data))
        + logpdf(Beta(6, 9), ξ + .5)
    )
end


"""
    neighborsMean(cellIndex, θ, F)

Compute the iGMRF neighbors influence over cell i for parameter θ.

# Arguments
- `cellIndex::Integer`: Index of current cell.
- `θ::DenseVector`: Values of the given parameter for all cells.
- `F::iGMRF`: iGMRF's structure.
"""
function neighborsMean(cellIndex::Integer, θ::DenseVector, F::iGMRF)
    return ((-F.G.W̄) * θ)[cellIndex] / F.G.W[cellIndex, cellIndex]
end


"""
    logapprox(θ, approxMarginals)

Return the log approximation density.
It is the sum of each log density of a cell.

# Arguments
- `θ::DenseVector`: Parameters [μ..., ϕ..., ξ...].
- `approxMarginals::Vector{<:Distribution}`: The marginal distribution of each cell.
"""
function logapprox(θ::DenseVector, approxMarginals::Vector{<:Distribution})

    M = length(approxMarginals) - 3;
    μ = θ[1:M];
    ϕ = θ[M+1:2*M];
    ξ = θ[2*M+1];
    κᵤ = θ[2*M+2];
    κᵥ = θ[2*M+3];

    return (
        sum([
            logpdf(approxMarginals[i], [μ[i], ϕ[i]])
            for i=1:M
        ]) 
        + logpdf(approxMarginals[M+1], ξ)
        + logpdf(approxMarginals[M+2], κᵤ)
        + logpdf(approxMarginals[M+3], κᵥ)
    );
end


"""
    compKappaParam(θ, var, F)

Compute Kappa's second parameter.

# Arguments :
- `θ::DenseVector`: Values of given parameter for all cells.
- `var::DenseVector`: Variance of given parameter for all cells.
- `F::iGMRF`: iGMRF's structure.
"""
function compKappaParam(θ::DenseVector, var::DenseVector, F::iGMRF)
    return (
        (
            sum(diag(F.G.W) .* (var .+ θ.^2))
            + θ' * F.G.W̄ * θ
        ) / 2
        + 0.01
    )
end