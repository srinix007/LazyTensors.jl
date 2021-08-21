module ParallelArrays

export LazyTensor, unwrap, LazyTensorStyle

struct LazyTensor{T,N,A <: DenseArray{T,N}} <: DenseArray{T,N}
    dat::A
end

LazyTensor(a::AbstractArray{T,N}) where {T,N} = LazyTensor{T,N,typeof(a)}(a)

Base.size(a::LazyTensor) = size(a.dat)
Base.getindex(a::LazyTensor, i...) = a.dat[i...]
Base.setindex!(a::LazyTensor, value, i...) = setindex!(a.dat, value, i...)
Base.eachindex(a::LazyTensor) = eachindex(a.dat)

struct LazyTensorStyle <: Base.Broadcast.BroadcastStyle end

Base.BroadcastStyle(::Type{<:LazyTensor}) = LazyTensorStyle()

Base.BroadcastStyle(::LazyTensorStyle, ::Broadcast.BroadcastStyle) = LazyTensorStyle() 

Base.materialize(bc::Broadcast.Broadcasted{LazyTensorStyle}) = bc

function Base.materialize!(dest, bc::Broadcast.Broadcasted{LazyTensorStyle})
    return LazyTensor(Base.materialize!(dest.dat, unwrap(bc)))
end

function unwrap(bc::Broadcast.Broadcasted{LazyTensorStyle})
    return Broadcast.broadcasted(bc.f, map(unwrap, bc.args)...)
end

unwrap(x) = x
unwrap(a::LazyTensor) = a.dat


end