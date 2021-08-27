using GPUArrays

# For GPUArrays dispatch to Base methods which will call the corresponding GPU methods


@inline threaded_materialize(bc::Broadcast.Broadcasted{<:AbstractGPUArrayStyle}) = LazyTensor(Base.materialize(bc))

@inline threaded_copyto!(dest, bc::Broadcast.Broadcasted{<:AbstractGPUArrayStyle}) = Base.materialize!(bc)

@inline serial_reduce(op, A::Broadcast.Broadcasted{<:AbstractGPUArrayStyle}, initval) = reduce(op, A)

@inline serial_reducedim(op, A::Broadcast.Broadcasted{<:AbstractGPUArrayStyle}, dims) = reduce(op, A, dims=dims)

@inline treduce(op, A::AbstractGPUArray, dims) = reduce(op, A, dims=dims)
@inline treduce(op,
                A::BroadcastArray{T,N,<:Broadcast.Broadcasted{<:AbstractGPUArrayStyle}},
                dims) where {T,N} = reduce(op, A.bc, dims=dims)

@inline threaded_reduce(op, A::AbstractGPUArray) = reduce(op, A)
@inline threaded_reduce(op,
    A::BroadcastArray{T,N,<:Broadcast.Broadcasted{<:AbstractGPUArrayStyle}}) where {T,N} = reduce(op, A.bc)