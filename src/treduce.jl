
tsum(A) = treduce(+, A)

tprod(A) = treduce(*, A)

function treduce(op, A) 
    initval = initfun(typeof(op))(eltype(A))
    threaded_reduce(op, A, initval)
end

function halve_dims(iter)
    len = length(iter)
    mid = len >> 1
    idx1 = view(iter, 1:mid)
    idx2 = view(iter, (mid + 1):len)
    return idx1, idx2
end

function halve(A::AbstractArray)
    idx1, idx2 = halve_dims(eachindex(A))
    return view(A, idx1), view(A, idx2)
end

function threaded_reduce(op, A, initval, nth=Threads.nthreads())

    if nth == 1
        return serial_reduce(op, A, initval)
    end

    A1, A2 = halve(A)
    nth2 = nth >> 1
    t = Threads.@spawn threaded_reduce(op, A1, initval, nth2)
    nth3 = nth - nth2
    op(threaded_reduce(op, A2, initval, nth3), fetch(t))
end










