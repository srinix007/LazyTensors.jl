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
    @testset "Halve size $s" for s in [(i, j, k) for i in 1:10, j in 1:10, k in 1:10]
        iter = CartesianIndices(s)
        ln = abs(prod(s))
        md = ln >> 1

       @test ParallelArrays.halve_dims(iter) == (CartesianIndices(s)[1:md], CartesianIndices(s)[(1 + md):end])
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

@testset "Halve dims Array" begin
        s = (4, 5, 4, 8, 13)
        iter = CartesianIndices(s)

    @test halve_dims(iter, (1,))[1] == CartesianIndices(s)[1:2, :, :, :, :]
    @test halve_dims(iter, (1,))[2] == CartesianIndices(s)[3:4, :, :, :, :]

    @test halve_dims(iter, (2,))[1] == CartesianIndices(s)[:, 1:2, :, :, :]
    @test halve_dims(iter, (2,))[2] == CartesianIndices(s)[:, 3:5, :, :, :]

    @test halve_dims(iter, (1, 2))[1] == CartesianIndices(s)[1:2, 1:2, :, :, :]
    @test halve_dims(iter, (1, 2))[2] == CartesianIndices(s)[3:4, 3:5, :, :, :]
end

