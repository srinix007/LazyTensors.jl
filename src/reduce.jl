
tsum(A) = treduce(+, A)
tsum(A,dims) = treduce(+, A, dims, zero(eltype(A)))

prod(A) = treduce(*, A)
prod(A, dims) = treduce(*, A, dims, one(eltype(A)))

treduce(op, A::AbstractArray) = threaded_reduce(op, A)

function treduce(op, A::AbstractArray{T}, dims, init::T) where {T}
    lred = prod(size(A)[i] for i in dims)
    lrem = length(A) - lred

    if lred < minimum(1000, lrem)
        threaded_reducedim_r(op, A, dims)
    else
        dest = Base.reducedim_initarray(A, dims, init)
        threaded_reducedim!(op, dest, A)
    end
end

function halve(A::AbstractArray)
    len = length(A)
    mid = len >> 1
    iter = eachindex(A)
    idx1 = view(iter, 1:mid)
    idx2 = view(iter, mid + 1:len)
    return view(A, idx1), view(A, idx2)
end

function threaded_reduce(op, A, nth=Threads.nthreads)

    if nth == 1
        dest = zero(eltype(A))
        Base.reduce(op, A)
        return dest
    end

    A1, A2 = halve(A)
    nth2 = nth >> 1
    t = Threads.@spawn threaded_reduce(op, A1, nth2)
    nth3 = nth - nth2
    op(threaded_reduce(op, A2, nth3), fetch(t))
end

function halve(A::AbstractArray, rdims) 
    ldims = size(A)
    dims = length(ldims)
    lrdims = map(x -> ldims[x], dims)
    mid = lrdims .÷ 2
    j = 0
    k = 0
    midx = Tuple((i in rdims) ? (1:mid[j += 1]) : Colon() for i in 1:dims)
    enx = Tuple((i in rdims) ? (mid[k += 1] + 1:ldims[k]) : Colon() for i in 1:dims)
    return view(A, midx), view(A, enx)
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

dest_dims(A, dims) = (i for i in 1:length(size(A)) if !(i in dims))

function threaded_reducedim!(op, dest, A, nth=Threads.nthreads())
    
    if nth == 1
        Base.reducedim!(op, dest, A)
        return nothing
    end

    D1, D2 = halve(D)
    nth2 = nth >> 1
    t = Threads.@spawn threaded_reducedim!(op, D1, A, nth2)
    nth3 = nth - nth2
    threaded_reducedim!(op, D2, A, nth3)
    wait(t)

    return nothing
end
