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
    @testset "Serial reduce $dims" for dims in (1, 2, 3, (1, 2), (2, 3), (1, 3))
        M = rand(100, 500, 100)
        v = rand(100)
        ML = LazyTensor(M)
        vl = LazyTensor(v)

        @test sum(ML .* vl, dims) ≈  dropdims(sum(M .* v, dims=dims), dims=dims)
        @test sum(ML .* 2.0, dims) ≈  dropdims(sum(M .* 2.0, dims=dims), dims=dims)
        @test sum(ML .* 2.0 .+ vl, dims) ≈  dropdims(sum(M .* 2.0 .+ v, dims=dims), dims=dims)

        R = Base.reducedim_initarray(M, dims, zero(eltype(M)))
        sum!(R, ML .* vl) 
        @test R ≈  sum(M .* v, dims=dims)
        R = Base.reducedim_initarray(M, dims, zero(eltype(M)))
        sum!(R, ML .* 2.0)
        @test R ≈ sum(M .* 2.0, dims=dims)
        R = Base.reducedim_initarray(M, dims, zero(eltype(M)))
        sum!(R, ML .* 2.0 .+ vl)
        @test R ≈ sum(M .* 2.0 .+ v, dims=dims)

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

@testset verbose = true "Serial reducedim $dims" for dims in (1, 2, 3, (1, 2), (2, 3), (1, 3))
        
        M = rand(100, 500, 400)
        v = rand(100)
        ML = LazyTensor(M)
        vl = LazyTensor(v)

        @testset "Array" begin
            @test ParallelArrays.serial_reducedim(+, M .* v, dims) ≈  dropdims(sum(M .* v, dims=dims), dims=dims)
            @test ParallelArrays.serial_reducedim(+, M .* 2.0, dims) ≈  dropdims(sum(M .* 2.0, dims=dims), dims=dims)
        end

        @testset "Broadcast" begin
            @test ParallelArrays.serial_reducedim(+, ML .* vl, dims) ≈  dropdims(sum(M .* v, dims=dims), dims=dims)
            @test ParallelArrays.serial_reducedim(+, ML .* 2.0, dims) ≈  dropdims(sum(M .* 2.0, dims=dims), dims=dims)
        end

end

@testset verbose = true "Parallel reducedim $dims" for dims in (1, 2, 3, (1, 2), (2, 3), (1, 3))
    if Threads.nthreads() > 1
        
        M = rand(100, 500, 400)
        v = rand(100)
        ML = LazyTensor(M)
        vl = LazyTensor(v)

        @testset "Array" begin
            @test tsum(M .* v, dims) ≈  dropdims(sum(M .* v, dims=dims), dims=dims)
            @test tsum(M .* 2.0, dims) ≈  dropdims(sum(M .* 2.0, dims=dims), dims=dims)
        end
        
        @testset "Broadcast" begin
            @test tsum(ML .* vl, dims) ≈  dropdims(sum(M .* v, dims=dims), dims=dims)
            @test tsum(ML .* 2.0, dims) ≈  dropdims(sum(M .* 2.0, dims=dims), dims=dims)
        end

        @testset "Array inplace" begin
            R = Base.reducedim_initarray(M, dims, zero(eltype(M)))
            tsum!(R, M .* v)
            @test R ≈  sum(M .* v, dims=dims)
            R = Base.reducedim_initarray(M, dims, zero(eltype(M)))
            tsum!(R, M .* 2.0)
            @test R ≈ sum(M .* 2.0, dims=dims)
        end

        @testset "Broadcast inplace" begin
            R = Base.reducedim_initarray(M, dims, zero(eltype(M)))
            tsum!(R, ML .* vl)
            @test R ≈  sum(M .* v, dims=dims)
            R = Base.reducedim_initarray(M, dims, zero(eltype(M)))
            tsum!(R, ML .* 2.0)
            @test R ≈ sum(M .* 2.0, dims=dims)
        end
        
        if ndims(M) in dims

            @testset "Aggregate Array" begin
                @test ParallelArrays.threaded_reducedim(+, M .* v, dims) ≈  dropdims(sum(M .* v, dims=dims), dims=dims)
                @test ParallelArrays.threaded_reducedim(+, M .* 2.0, dims) ≈  dropdims(sum(M .* 2.0, dims=dims), dims=dims)
            end
            
            @testset "Aggregate Broadcast" begin
                @test ParallelArrays.threaded_reducedim(+, ML .* vl, dims) ≈  dropdims(sum(M .* v, dims=dims), dims=dims)
                @test ParallelArrays.threaded_reducedim(+, ML .* 2.0, dims) ≈  dropdims(sum(M .* 2.0, dims=dims), dims=dims)
            end
        end


    end
end