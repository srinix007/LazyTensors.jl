
tsum(A) = treduce(+, A)

tprod(A) = treduce(*, A)

@inline treduce(op, A::AbstractArray) = threaded_reduce(op, A)

function threaded_reduce(op, A, st=1, en=length(A), nth=Threads.nthreads())

    if nth == 1
        initval = initfun(typeof(op))(eltype(A))
        return serial_reduce(op, A, initval, st, en)
    end

    mid = (st + en) >> 1
    nth2 = nth >> 1
    t = Threads.@spawn threaded_reduce(op, A, st, mid, nth2)
    nth3 = nth - nth2
    op(threaded_reduce(op, A, mid + 1, en, nth3), fetch(t))
end










