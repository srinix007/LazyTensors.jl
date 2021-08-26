using ParallelArrays
using Test

@testset "Serial Collect" begin
    M = rand(100, 100)
    v = rand(100)
    ML = LazyTensor(M)
    vl = LazyTensor(v)

    @test  ev(ML .* 3.0) ≈ M .* 3.0
    @test  ev(ML .* vl) ≈ M .* v
    @test ev(ML .* sin.(cos.(vl))) ≈ M .* sin.(cos.(v))
end

@testset "Parallel Collect" begin
    if Threads.nthreads() > 1

        M = rand(1000, 1000)
        v = rand(1000)
        ML = LazyTensor(M)
        vl = LazyTensor(v)

        @test  pev(ML .* 3.0) ≈ M .* 3.0
        @test  pev(ML .* vl) ≈ M .* v
        @test pev(ML .* sin.(cos.(vl))) ≈ M .* sin.(cos.(v))
        @test pev(ML .* [sin.(cos.(vl))] .+ [tan.(vl)]) ≈ M .* sin.(cos.(v)) .+ tan.(v)
    end
end

@testset "Serial Reduce" begin
    M = rand(1000, 500)
    v = rand(1000)
    ML = LazyTensor(M)
    vl = LazyTensor(v)

    @test sum(ML .* vl) ≈  sum(M .* v)
    @test sum(ML .* 2.0) ≈  sum(M .* 2.0)
    @test sum(ML .* 2.0 .+ vl) ≈  sum(M .* 2.0 .+ v)

end

@testset verbose = true "Serial Partial reduce" begin
    @testset "Serial reduce $dims" for dims in 1:2 
        M = rand(1000, 500)
        v = rand(1000)
        ML = LazyTensor(M)
        vl = LazyTensor(v)

        @test sum(ML .* vl, dims) ≈  sum(M .* vl, dims=dims)
        @test sum(ML .* vl, dims) ≈  sum(M .* vl, dims=dims)
        @test sum(ML .* 2.0, dims) ≈  sum(M .* 2.0, dims=dims)
        @test sum(ML .* 2.0 .+ vl, dims) ≈  sum(M .* 2.0 .+ v, dims=dims)

    end
end

@testset "Halve Array" begin
    @testset "Cartesian Indices" begin
        iter = [CartesianIndices(s) for s in [(10, 9, 8, 4), (9, 65, 32, 23), (56, 33, 456)]]

       @test halve_dims(iter[1]) == ([1:5, 1:9, 1:8, 1:4], [6:10, 1:9, 1:8, 1:4])
       @test halve_dims(iter[2]) == ([1:9, 1:32, 1:32, 1:23], [1:9, 33:65, 1:32, 1:23])
       @test halve_dims(iter[3]) == ([1:56, 1:33, 1:228], [1:56, 1:33, 229:456])
    end
end

@testset verbose = true "Parallel Reduce" begin
    if Threads.nthreads() > 1

        M = rand(1000, 500)
        v = rand(1000)
        ML = LazyTensor(M)
        vl = LazyTensor(v)

        @testset "Array" begin
            @test tsum(M .* v) ≈  sum(M .* v)
            @test tsum(M .* 2.0) ≈  sum(M .* 2.0)
            @test tsum(M .* 2.0 .+ v) ≈  sum(M .* 2.0 .+ v)
        end

        @testset "Broadcast" begin
            @test tsum(ML .* vl) ≈  sum(M .* v)
            @test tsum(ML .* 2.0) ≈  sum(M .* 2.0)
            @test tsum(ML .* 2.0 .+ vl) ≈  sum(M .* 2.0 .+ v)
        end
    end
end

@testset verbose = true "Parallel reducedim $dims" for dims in (1, 2, 3, (1, 2), (2, 3), (1, 3))
    if Threads.nthreads() > 1
        
        M = rand(100, 500, 400)
        v = rand(100)
        ML = LazyTensor(M)
        vl = LazyTensor(v)

    @testset "Split reduce" begin

        @testset "Array" begin
            @test ParallelArrays.threaded_reducedim_r(+, M .* v, dims) ≈  sum(M .* v, dims=dims)
            @test ParallelArrays.threaded_reducedim_r(+, M .* 2.0, dims) ≈  sum(M .* 2.0, dims=dims)
        end

        @testset "Broadcast" begin
            @test ParallelArrays.threaded_reducedim_r(+, ML .* vl, dims) ≈  sum(M .* v, dims=dims)
            @test ParallelArrays.threaded_reducedim_r(+, ML .* 2.0, dims) ≈  sum(M .* 2.0, dims=dims)
        end

    end

    @testset "Split map" begin

        R = Base.reducedim_initarray(M, dims, 0.0)
        mdims = ParallelArrays.dest_dims(M, dims)

        @testset "Array" begin
            ParallelArrays.threaded_reducedim!(+, R, M .* v, dims, mdims)
            @test R ≈  sum(M .* v, dims=dims)

            R .= 0.0
            ParallelArrays.threaded_reducedim!(+, R, M .* 2.0, dims, mdims) 
            @test R ≈  sum(M .* 2.0, dims=dims)
        end

        @testset "Broadcast" begin
            R .= 0.0
            ParallelArrays.threaded_reducedim!(+, R, ML .* vl, dims, mdims)
            @test R ≈  sum(M .* v, dims=dims)

            R .= 0.0
            ParallelArrays.threaded_reducedim!(+, R, ML .* 2.0, dims, mdims)
            @test R ≈  sum(M .* 2.0, dims=dims)
        end
        
    end
end
end