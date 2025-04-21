using Test, Suppressor

include("../src/cavi.jl");
include("ressources/cavi.jl");

@testset "cavi.jl" begin

    @testset "buildVar(components)" begin

        components = [1, 2, 3, 4];
        @test buildVar(components)[1, 2] == 2;

    end


    @testset "compApproxMarginals!(approxMarginals, traces; caviCounter, spatialScheme)" begin

        traces = cAM!ressource.traces;
        caviCounter = cAM!ressource.caviCounter;
        spatialScheme = cAM!ressource.spatialScheme;
        M = spatialScheme[:M];

        approxMarginals = Vector{Distribution}(undef, M+3);

        compApproxMarginals!(approxMarginals, traces, caviCounter=caviCounter, spatialScheme=spatialScheme);

        @test all(approxMarginals .!= undef);
        @test approxMarginals[M+3] == Gamma(1, .5);
    end


    @testset "estimateKappa(variant; traces, caviCounter)" begin
        
        traces = Dict(
            :kappaUparams => [0 1; 0 2],
        );
        caviCounter = Dict(
            :iter => 2,
        );

        @test estimateKappa("u", traces=traces, caviCounter=caviCounter) ≈ .5;
    end


    @testset "updateParams!(traces, caviCounter, spatialScheme)" begin
        
        traces = uP!ressource.traces;
        caviCounter = uP!ressource.caviCounter;
        spatialScheme = uP!ressource.spatialScheme;

        updateParams!(traces, caviCounter, spatialScheme);

        @test all(traces[:muMean][:, 2] != 0.0);
        @test all(traces[:phiMean][:, 2] != 0.0);
        @test all(traces[:cellVar][:, 2, 1] != 0.0);
        @test traces[:xiMean][2] != 0.0;
        @test traces[:kappaUparams][2, 2] != 1.0;
        @test traces[:kappaVparams][2, 2] != 1.0;

    end

    @testset "runIter!(traces; caviCounter, spatialScheme)" begin
        
        traces = runIter!ressource.traces;
        caviCounter = runIter!ressource.caviCounter;
        spatialScheme = runIter!ressource.spatialScheme;
        M = spatialScheme[:M];

        @suppress begin
            runIter!(traces, caviCounter=caviCounter, spatialScheme=spatialScheme);
        end

        @test size(traces[:muMean]) == (M, 2);
        @test size(traces[:phiMean]) == (M, 2);
        @test size(traces[:cellVar]) == (4, M, 2);
        @test !any(isnan.(traces[:cellVar][:, :, 2]));
        @test size(traces[:xiMean]) == (2,);
        @test size(traces[:kappaUparams]) == (2, 2);
        @test size(traces[:kappaVparams]) == (2, 2);


    end


    @testset "runCAVI(nEpochMax, epochSize, initialValues, spatialScheme, ϵ)" begin
        
        nEpochMax = runCAVIressource[:nEpochMax];
        epochSize = runCAVIressource[:epochSize];
        initialValues = runCAVIressource[:initialValues];
        spatialScheme = runCAVIressource[:spatialScheme];

        res = @suppress begin
            runCAVI(nEpochMax, epochSize, initialValues, spatialScheme);
        end
        
        @test length(res.MCKL) == nEpochMax * epochSize + 1;
        @test size(res.traces[:muMean]) == (M, nEpochMax * epochSize + 1);
        @test size(res.traces[:cellVar]) == (4, M, nEpochMax * epochSize + 1);


    end
end