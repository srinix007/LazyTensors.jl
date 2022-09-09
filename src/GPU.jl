using GPUArrays

# For GPUArrays dispatch to Base methods which will call the corresponding GPU methods


@inline threaded_materialize(bc::Broadcast.Broadcasted{<:AbstractGPUArrayStyle}) = Base.materialize(bc)

@inline threaded_copyto!(dest, bc::Broadcast.Broadcasted{<:AbstractGPUArrayStyle}) = Base.materialize!(dest, bc)

@inline serial_reduce(op, A::Broadcast.Broadcasted{<:AbstractGPUArrayStyle}, initval) = reduce(op, A)

@inline serial_reducedim(op, A::Broadcast.Broadcasted{<:AbstractGPUArrayStyle}, dims) = reduce(op, A, dims=dims)

@inline treduce(op, A::AbstractGPUArray, dims) = reduce(op, A, dims=dims)
@inline treduce(op,
    A::BroadcastArray{T,N,<:Broadcast.Broadcasted{<:AbstractGPUArrayStyle}},
    dims) where {T,N} = reduce(op, A.bc, dims=dims)

@inline threaded_reduce(op, A::AbstractGPUArray) = reduce(op, A)
@inline threaded_reduce(op,
    A::BroadcastArray{T,N,<:Broadcast.Broadcasted{<:AbstractGPUArrayStyle}}) where {T,N} = reduce(op, A.bc)

function Base.sum!(R::AbstractGPUArray,
    A::BroadcastArray{T,N,<:Broadcast.Broadcasted{<:AbstractGPUArrayStyle}}, dims) where {T,N}
    Rs = reshape(R, Base.reduced_indices(A, dims))
    Base.mapreducedim!(identity, +, Rs, A.bc)
    return nothing
end

function Base.sum!(R::AbstractGPUArray, A::AbstractGPUArray, dims)
    Rs = reshape(R, Base.reduced_indices(A, dims))
    Base.sum!(Rs, A)
    return nothing
end

function Base.prod!(R::AbstractGPUArray,
    A::BroadcastArray{T,N,<:Broadcast.Broadcasted{<:AbstractGPUArrayStyle}}, dims) where {T,N}
    Rs = reshape(R, Base.reduced_indices(A, dims))
    Base.mapreducedim!(identity, *, Rs, A.bc)
    return nothing
end

function Base.prod!(R::AbstractGPUArray, A::AbstractGPUArray, dims)
    Rs = reshape(R, Base.reduced_indices(A, dims))
    prod!(Rs, A)
    return nothing
end