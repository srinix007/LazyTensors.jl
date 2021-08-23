using ParallelArrays
using Test

@testset "Serial Collect" begin
    M = rand(200, 100)
    v = rand(100)
    ML = LazyTensor(M)
    vl = LazyTensor(v)

    @test  ev(ML .* 3.0) ≈ M .* 3.0
    @test  ev(ML .* vl) ≈ M .* v
    @test ev(ML .* sin.(cos.(vl))) ≈ M .* sin.(cos.(v))
end
