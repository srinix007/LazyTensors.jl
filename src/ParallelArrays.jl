module ParallelArrays

export LazyTensor, unwrap, LazyTensorStyle, pev, ev, tsum, tprod, treduce

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

function unwrap(bc::Broadcast.Broadcasted{LazyTensorStyle})
    return Broadcast.broadcasted(bc.f, map(unwrap, bc.args)...)
end

unwrap(x) = x
unwrap(a::LazyTensor) = a.dat

include("collect.jl")
include("reduce.jl")

end