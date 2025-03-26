using Test, GMRF, SpecialFunctions, Distributions, LinearAlgebra

include("../src/model.jl");

@testset "model.jl" begin
    
    @testset "logposterior(θ; Fmu, Fphi, data)" begin
        
        M₁ = 2;
        M₂ = 2;
        Fmu = iGMRF(M₁, M₂, 1, 1.0);
        Fphi = iGMRF(M₁, M₂, 1, 1.0);
        data = [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ];

        μ = zeros(4);
        ϕ = zeros(4);
        ξ = 0.0;
        κᵤ = 10.0;
        κᵥ = 100.0;

        @test logposterior([μ..., ϕ..., ξ, κᵤ, κᵥ], Fmu=Fmu, Fphi=Fphi, data=data) ≈ (
            - 8 - 8 / ℯ 
            + 1.5 * (log(10) + log(100))
            - log(100) - .1
            - log(100) - 1
            + loggamma(15) - loggamma(6) - loggamma(9) + 13 * log(.5)
        )
    end


    @testset "neighborsMean(cellIndex, θ, F)" begin
        
        F = iGMRF(2, 2, 1, 1.0); # Peu importe le kappa
        θ = [1.0, 2.0, 3.0, 5.0];

        @test neighborsMean(1, θ, F) ≈ 2.5;
        @test neighborsMean(2, θ, F) ≈ 3.0;
    end


    @testset "celllogfullconditional(i, θi; μ̄i, ϕ̄1, Fmu, Fphi, data)" begin
        
        M₁ = 2;
        M₂ = 2;
        M = M₁ * M₂;

        i = 1;
        θi = [1.0, 0.0];
        ξ = 0.0;
        μ̄i = 0.0;
        ϕ̄i = 0.0;
        κ̂ᵤ = 10.0;
        κ̂ᵥ = 100.0;
        Fmu = iGMRF(M₁, M₂, 1, 1.0);
        Fphi = iGMRF(M₁, M₂, 1, 1.0);
        data = [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ];

        @test celllogfullconditional(i, θi, ξ=ξ, μ̄i=μ̄i, ϕ̄i=ϕ̄i, κ̂ᵤ=κ̂ᵤ, κ̂ᵥ=κ̂ᵥ, Fmu=Fmu, Fphi=Fphi, data=data) ≈ (
            - 2 # loglike GEV
            - .5 * log(2*pi / 20) - 10 # Priori de mu
            - .5 * log(2*pi / 200) # Priori de phi
        )
    end
    
    
    @testset "xilogfullconditional(ξ; μ, ϕ, data)" begin
        
        M₁ = 2;
        M₂ = 2;
        M = M₁ * M₂;

        ξ = 0.0
        Eμ = ones(4);
        Eϕ = zeros(4);
        varϕ = fill(2, 4);
        data = [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ];

        @test xilogfullconditional(ξ, Eμ=Eμ, Eϕ=Eϕ, varϕ=varϕ, data=data) ≈ (
            - 16 # loglike GEV
            + loggamma(15) - loggamma(6) - loggamma(9) + 13 * log(.5) # Priori de xi
        )
    end


    @testset "logapprox(θ, approxMarginals)" begin
        
        θ = [i for i = 1:7];

        approxMarginals = [
            MvNormal(zeros(2), I),
            MvNormal(zeros(2), I),
            Normal(0.0, 1.0),
            Gamma(1.0, 2.0),
            Gamma(1.0, 7.0),
        ];

        @test logapprox(θ, approxMarginals) ≈ (
            - log(2*pi) - 5 # Marginale 1ere cellule
            - log(2*pi) - 10 # Marginale 2eme cellule
            - log(2*pi) / 2 - 25 / 2 # Marginale xi
            - log(2) - 3 # Marginale kappa_u
            - log(7) - 1 # Marginale kappa_v
        )
    end


    @testset "compKappaParam(θ, var, F)" begin
        
        M₁ = 2;
        M₂ = 2;
        M = M₁ * M₂;

        F = iGMRF(M₁, M₂, 1, 1.0);
        θ = ones(M);
        var = fill(2.0, M);

        @test compKappaParam(θ, var, F) ≈ 8.01;

    end
end