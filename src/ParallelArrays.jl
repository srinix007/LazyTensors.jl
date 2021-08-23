module ParallelArrays

export LazyTensor, unwrap, LazyTensorStyle, pev, ev

struct LazyTensor{T,N,A <: AbstractArray{T,N}} <: DenseArray{T,N}
    dat::A
end

LazyTensor(a::AbstractArray{T,N}) where {T,N} = LazyTensor{T,N,typeof(a)}(a)

Base.parent(a::LazyTensor) = a.dat

Base.size(a::LazyTensor) = size(a.dat)
Base.getindex(a::LazyTensor, i...) = a.dat[i...]
Base.setindex!(a::LazyTensor, value, i...) = setindex!(a.dat, value, i...)
Base.eachindex(a::LazyTensor) = eachindex(a.dat)
Base.strides(a::LazyTensor) = strides(a.dat)


struct LazyTensorStyle <: Base.Broadcast.BroadcastStyle end

Base.BroadcastStyle(::Type{<:LazyTensor}) = LazyTensorStyle()

Base.BroadcastStyle(::LazyTensorStyle, ::Broadcast.BroadcastStyle) = LazyTensorStyle() 

Base.materialize(bc::Broadcast.Broadcasted{LazyTensorStyle}) = bc

"""
    unwrap(bc::Broadcast.Broadcasted{LazyTensorStyle})

Unwraps all LazyTensor objects to its parent in the Broadcast tree.

# Examples

```jldoctest
julia> L = LazyTensor(rand(2,2));

julia> typeof(L)
LazyTensor{Float64, 2, Matrix{Float64}}

julia> typeof(L .* 2.0)
Base.Broadcast.Broadcasted{LazyTensorStyle, Nothing, typeof(*), Tuple{LazyTensor{Float64, 2, Matrix{Float64}}, Float64}}

julia> typeof(unwrap(bc))
Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{2}, Nothing, typeof(*), Tuple{Matrix{Float64}, Float64}}
```

"""
function unwrap(bc::Broadcast.Broadcasted{LazyTensorStyle})
    return Broadcast.broadcasted(bc.f, map(unwrap, bc.args)...)
end

unwrap(x) = x
unwrap(a::LazyTensor) = a.dat

include("collect.jl")

end