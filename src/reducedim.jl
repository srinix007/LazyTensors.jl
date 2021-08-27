
@inline Base._sum(A::BroadcastArray, dims) = sum(A, dims)
@inline Base._prod(A::BroadcastArray, dims) = prod(A, dims)

@inline Base.sum(A::BroadcastArray, dims) = reduce(+, A, dims)
@inline Base.prod(A::BroadcastArray, dims) = reduce(*, A, dims)


@inline Base.reduce(op::Function, A::BroadcastArray, dims) = serial_reducedim(op, unwrap(A), dims)

function serial_reducedim(op, A, dims)
    inval = initfun(typeof(op))(eltype(A))
    R = Base.reducedim_initarray(A, dims, inval)
    serial_reducedim!(op, R, A)
    return R
end

function serial_reducedim!(op, R, A)

    isempty(A) && return R

    indsAt, indsRt = Base.safe_tail(axes(A)), Base.safe_tail(axes(R))
    keep, Idefault = Broadcast.shapeindexer(indsRt)

    if Base.reducedim1(R, A)
        i1 = first(Base.axes1(R))
        @inbounds for IA in CartesianIndices(indsAt)
            IR = Broadcast.newindex(IA, keep, Idefault)
            r = R[i1,IR]
            @inbounds @simd for i in axes(A, 1)
                @inbounds r = op(r, A[i, IA])
            end
            R[i1,IR] = r
        end
    else
        @inbounds for IA in CartesianIndices(indsAt)
            IR = Broadcast.newindex(IA, keep, Idefault)
            @inbounds @simd for i in axes(A, 1)
                @inbounds R[i,IR] = op(R[i,IR], A[i,IA])
            end
        end
    end
    return R 
end