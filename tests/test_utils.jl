using Test, Optim, Distributions, LinearAlgebra
using Distributions:loglikelihood

include("../src/utils.jl");

@testset "utils.jl" begin
    
    @testset "findMode(f, x0)" begin
        
        f(x::DenseVector) = -(x[1]-1)^2 - (x[2]-1)^2;

        mode = findMode(f, [0.0, 0.0]);

        @test (mode[1] - 1.0) < .0001;
        @test (mode[2] - 1.0) < .0001;

    end


    @testset "fisherVar(f, x)" begin
        
        m = [0.0, 0.0, 0.0];
        Σ = diagm([1.0, 2.0, 3.0]);

        logf(x::DenseVector) = logpdf(MvNormal(m, Σ), x);

        @test fisherVar(logf, m) ≈ Σ;

    end


    @testset "fisherVar(f, x0)" begin
        
        μ = 0.0;
        σ = 2.0;

        logf(x::Real) = logpdf(Normal(μ, σ), x);

        @test fisherVar(logf, 0.0) ≈ σ^2;

    end


    @testset "flatten(mat)" begin
        
        mat = [1 2 3; 4 5 6; 7 8 9];

        vector = flatten(mat);

        @test vector[2] == 2;
        @test vector[6] == 6;
        @test length(vector) == 9;

    end


    @testset "vecToMatrix(vector)" begin
        
        vector = [1, 2, 3, 4, 5, 6, 7, 8, 9];

        mat = vecToMatrix(vector);

        @test mat[1, 2] == 2;
        @test mat[2, 3] == 6;
        @test size(mat) == (3, 3);

    end
    
    
    @testset "tensToMat(tensor)" begin
        
        tensor = reshape([i for i=1:81], (3, 3, 9));

        mat = tensToMat(tensor);

        @test length(mat[:, 1]) == 27;
        @test mat[:, 1] == [
            [(1.0 + (k-1) * 9) for k = 1:9]...,
            [(2.0 + (k-1) * 9) for k = 1:9]...,
            [(3.0 + (k-1) * 9) for k = 1:9]...,
        ];

    end


    @testset "fastInv(M)" begin
        
        M = [1 0 0; 0 2 0; 0 0 3];

        invM = fastInv(M);

        @test invM[1, 2] == 0.0;
        @test invM[2, 2] ≈ 1/2;
        @test invM[3, 3] ≈ 1/3;

    end
end