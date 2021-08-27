
tsum(A, dims) = treduce(+, A, dims)
tprod(A, dims) = treduce(*, A, dims)

function treduce(op, A::AbstractArray, dims)
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

@inline treduce(op, A::AbstractGPUArray, dims) = reduce(op, A, dims=dims)
@inline treduce(op,
                A::BroadcastArray{T,N,<:Broadcast.Broadcasted{<:AbstractGPUArrayStyle}},
                dims) where {T,N} = reduce(op, A.bc, dims=dims)

function halve_dims(s, rdims) 
    maxid = argmax([s[i] for i in rdims])
    maxid = rdims[maxid]
    idx = [1:i for i in s]
    mid = s[maxid] >> 1
    idx[maxid] = 1:mid
    midx = copy(idx)
    idx[maxid] = (mid+1):s[maxid]
    return midx, idx
end

function halve(A::AbstractArray, rdims)
    idx1, idx2 = halve_dims(size(A), rdims)
    return view(A, idx1...), view(A, idx2...)
end

function threaded_reducedim_r(op, A, rdims, nth=Threads.nthreads())
    
    if nth == 1
        dest = serial_reducedim(op, A, rdims)
        return dest
    end

    A1, A2 = halve(A, rdims)
    nth2 = nth >> 1
    t = Threads.@spawn threaded_reducedim_r(op, A1, rdims, nth2)
    nth3 = nth - nth2
    op.(threaded_reducedim_r(op, A2, rdims, nth3), fetch(t))
end

dest_dims(A, dims) = Tuple(i for i in 1:length(size(A)) if !(i in dims))

function threaded_reducedim!(op, dest, A, rdims, dest_dims, nth=Threads.nthreads())
    
    if nth == 1
        serial_reducedim!(op, dest, A)
        return nothing
    end

    D1, D2 = halve(dest, dest_dims)
    A1, A2 = halve(A, dest_dims)
    nth2 = nth >> 1
    t = Threads.@spawn threaded_reducedim!(op, D1, A1, rdims, dest_dims, nth2)
    nth3 = nth - nth2
    threaded_reducedim!(op, D2, A2, rdims, dest_dims, nth3)
    wait(t)

    return nothing
end
