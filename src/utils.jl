using Optim, ForwardDiff, LinearAlgebra

"""
    findMode(f, x0)

Find the mode of a given density f from starting point x0.

# Arguments
- `f::Function`: Functional form of the density.
- `x0::DenseVector`: Starting point (must be a vector).
"""
function findMode(f::Function, x0::DenseVector)
    F(x::DenseVector) = -f(x);
    mode = optimize(F, x0);
    return Optim.minimizer(mode)
end


"""
    findMode(f, x0)

Find the mode of a given density f from starting point x0.

# Arguments
- `f::Function`: Functional form of the density.
- `x0::Real`: Starting point (must be a vector).
"""
function findMode(f::Function, x0::Real)
    F(x::DenseVector) = -f(x[1]);
    mode = optimize(F, [x0]);
    return Optim.minimizer(mode)
end


"""
    fisherVar(f, x)

Compute the covariance matrix derived from the Fisher information.

# Arguments
- `logf::Function`: Log-functional form of the density.
- `x::DenseVector`: Where to compute the Fisher info (usually the mode).
"""
function fisherVar(logf::Function, x::DenseVector)
    return - fastInv(ForwardDiff.hessian(logf, x))
end


"""
    fisherVar(f, x0)

Compute the one-dimensional variance derived from the Fisher information.

# Arguments
- `logf::Function`: Log-functional form of the density.
- `x0::Real`: Where to compute the Fisher info (usually the mode).
"""
function fisherVar(logf::Function, x0::Real)
    return - 1 / ForwardDiff.derivative(x -> ForwardDiff.derivative(logf, x), x0)
end


"""
    flatten(mat)

Return a vector containing all components of a given matrix.

# Arguments
- `mat::Matrix{<:Real}`: Matrix to flatten.
"""
function flatten(mat::Matrix{<:Real})
    return mat'[:]
end


"""
    vecToMatrix(vector)

Create a nxn matrix from a given n²-element vector.
"""
function vecToMatrix(vector::DenseVector)
    
    n = Integer(sqrt(length(vector)));

    return round.(
        Matrix(
            reshape(vector, (n, n))'
        ),
        digits=7,
    )
end


"""
    tensToMat(tensor)

Reshape a (G x N x M) tensor to create a (G*M x N) matrix.
Goal : generate a vector [μ..., ϕ...] N times.
"""
function tensToMat(tensor::Array{<:Real, 3})
    
    G, N, M = size(tensor);
    mat = zeros(G * M, N);

    for n = 1:N
        mat[:, n] = vcat([tensor[g, n, :] for g=1:G]...);
    end
    
    return mat
end


"""
    fastInv(M)

fast inversion of matrix M.
"""
function fastInv(M::Matrix)
    return M \ I
end