
tsum(A) = treduce(+, A)

tprod(A) = treduce(*, A)

@inline treduce(op, A::AbstractArray) = threaded_reduce(op, A)

function halve_dims(iter::AbstractUnitRange)
    len = length(iter)
    mid = len >> 1
    return (1:mid,), ((1 + mid):len,)
end

function halve_dims(iter::CartesianIndices)
    s = size(iter)
    maxid = argmax(s)
    idx = [1:i for i in s]
    mid = s[maxid] >> 1
    idx[maxid] = 1:mid
    midx = copy(idx)
    idx[maxid] = (mid + 1):s[maxid]
    return midx, idx
end

function halve(A::AbstractArray)
    idx1, idx2 = halve_dims(eachindex(A))
    return view(A, idx1...), view(A, idx2...)
end

function threaded_reduce(op, A, nth=Threads.nthreads())

    if nth == 1
        return reduce(op, A)
    end

    A1, A2 = halve(A)
    nth2 = nth >> 1
    t = Threads.@spawn threaded_reduce(op, A1, nth2)
    nth3 = nth - nth2
    op(threaded_reduce(op, A2, nth3), fetch(t))
end










