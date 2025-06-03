include("utils.jl");
include("model.jl");

"""
    MonteCarloKL(logTargetDensity, approxMarginals)

Compute the convergence criterion with current hyper-parameters.

# Arguments
TBD
"""
function MonteCarloKL(logTargetDensity::Function, approxMarginals::Vector{<:Distribution})
    
    N = 1000;
    supp = generateApproxSample(approxMarginals, N);

    logApproxDensity(θ::DenseVector) = logapprox(θ, approxMarginals);

    logTarget = evaluateLogMvDensity(x -> logTargetDensity(x), supp);
    logApprox = evaluateLogMvDensity(x -> logApproxDensity(x), supp);
    
    return sum(logApprox .- logTarget) / N
end;


"""
    generateApproxSample(approxMarginals, N)

Draw samples from the approximating distribution with current hyper-parameters.

Use the mean-field approximation by generating each variable independantly.
Will be used to compute KL divergence.

# Arguments :
- `approxMarginals::Vector{Distribution}`: The marginal distribution of each parameter.
- `N::Integer`: Sample size (the same for all variables).
"""
function generateApproxSample(approxMarginals::Vector{<:Distribution}, N::Integer)
    
    m = length(approxMarginals) - 3;
    cellSupp = zeros(2, N, m);
    xiSupp = zeros(1, N);
    kappaUsupp = zeros(1, N);
    kappaVsupp = zeros(1, N);

    for k = 1:m
        cellSupp[:, :, k] = rand(approxMarginals[k], N);
    end
    xiSupp[1, :] = rand(approxMarginals[m+1], N);
    kappaUsupp[1, :] = rand(approxMarginals[m+2], N);
    kappaVsupp[1, :] = rand(approxMarginals[m+3], N);
    
    return vcat(tensToMat(cellSupp), xiSupp, kappaUsupp, kappaVsupp);
end


"""
    evaluateLogMvDensity(f, supp)

Evaluate a log multivariate density over a given set of vectors.

# Arguments :
- `f::Function`: Log multivariate density function to evaluate.
- `supp::Matrix{<:Real}`: Set of n p-arrays stored in a (p x n) matrix.
"""
function evaluateLogMvDensity(f::Function, supp::Matrix{<:Real})
    return vec(mapslices(f, supp, dims=1))
end