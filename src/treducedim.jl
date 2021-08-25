
tsum(A, dims) = treduce(+, A, dims)
tprod(A, dims) = treduce(*, A, dims)

function treduce(op, A::AbstractArray{T}, dims) where {T}
    rdims = Tuple(dims)
    lred = prod(size(A)[i] for i in rdims)
    lrem = length(A) - lred

    if lred < min(1000, lrem)
        res = threaded_reducedim_r(op, A, rdims)
    else
        initval = initfun(typeof(op))(eltype(A))
        dest = Base.reducedim_initarray(A, rdims, initval)
        threaded_reducedim!(op, dest, A, rdims, dest_dims(A, rdims))
        res = dest
    end
    return dest
end

function halve_dims(iter, rdims) 
    ldims = size(iter)
    maxdim = rdims[argmax([ldims[i] for i in rdims])]
    mid = ldims[maxdim] >> 1
    midx = Tuple( (i == maxdim) ? (1:mid) : Colon() for i in 1:ndims(iter))
    enx = Tuple( (i == maxdim) ? ((1+mid):ldims[i]) : Colon() for i in 1:ndims(iter))
    return view(iter, midx...), view(iter, enx...)
end

function halve(A::AbstractArray, rdims)
    idx1, idx2 = halve_dims(CartesianIndices(A), rdims)
    return view(A, idx1), view(A, idx2)
end

function threaded_reducedim_r(op, A, rdims, nth=Threads.nthreads())
    
    if nth == 1
        dest = reduce(op, A, dims=rdims)
        return dest
    end

    A1, A2 = halve(A, rdims)
    nth2 = nth >> 1
    t = Threads.@spawn threaded_reducedim_r(op, A1, rdims, nth2)
    nth3 = nth - nth2
    op.(threaded_reducedim_r(op, A2, rdims, nth3), fetch(t))
end

dest_dims(A, dims) = Tuple(i for i in 1:length(size(A)) if !(i in dims))

function threaded_reducedim!(op, dest, A, dims, rdims, nth=Threads.nthreads())
    
    if nth == 1
        serial_reducedim!(op, dest, A)
        return nothing
    end

    D1, D2 = halve(dest, rdims)
    A1, A2 = halve(A, rdims)
    nth2 = nth >> 1
    t = Threads.@spawn threaded_reducedim!(op, D1, A1, dims, rdims, nth2)
    nth3 = nth - nth2
    threaded_reducedim!(op, D2, A2, dims, rdims, nth3)
    wait(t)

    return nothing
end
