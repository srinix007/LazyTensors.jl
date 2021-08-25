
@inline initfun(::Type{typeof(+)}) = zero
@inline initfun(::Type{typeof(*)}) = one

@inline function Base.reduce(op::Function, A::BroadcastArray) 
    initval = initfun(typeof(op))(eltype(A))
    return serial_reduce(op, unwrap(A), initval)
end

function serial_reduce(op, A, initval, iter=eachindex(A))
    s = initval
    @inbounds @simd for i in iter
        @inbounds s += A[i]
    end
    return s
end

@inline Base.sum(A::BroadcastArray) = reduce(+, A)
@inline Base.prod(A::BroadcastArray) = reduce(*, A)