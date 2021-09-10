
tsum(A, dims) = treduce(+, A, dims)
tprod(A, dims) = treduce(*, A, dims)

function treduce(op, A::AbstractArray, dims, minsize=10000)
    sa = size(A)
    sr = prod(sa[i] for i in dims)
    sr = length(A) - sr
    sr = Threads.nthreads() * sr

    if ndims(A) in dims && sr < minsize
        return threaded_reducedim(op, A, dims)
    else
        initval = initfun(typeof(op))(eltype(A))
        R = Base.reducedim_initarray(A, dims, initval)
        threaded_reducedim!(op, R, A)
        return R
    end
end

function halve_dest(R, A)
    rdims = size(R)
    adims = size(A)

    ldim = findlast(x -> x != 1, size(R))

    mid = rdims[ldim] >> 1

    rmidx = [1:i for i in rdims]
    renx = [1:i for i in rdims]

    amidx = [1:i for i in adims]
    aenx = [1:i for i in adims]

    rmidx[ldim] = 1:mid
    renx[ldim] = (mid + 1):rdims[ldim]

    amidx[ldim] = 1:mid
    aenx[ldim] = (mid + 1):adims[ldim]

    return view(R, rmidx...), view(R, renx...), view(A, amidx...), view(A, aenx...)
end

function halve_last(A)
    adims = size(A)

    amidx = [1:i for i in adims]
    aenx = [1:i for i in adims]
    
    mid = adims[end] >> 1

    amidx[end] = 1:mid
    aenx[end] = (mid + 1):adims[end]

    return view(A, amidx...), view(A, aenx...)
end

function threaded_reducedim!(op, R, A, nth=Threads.nthreads())

    if nth == 1
        serial_reducedim!(op, R, A)
        return nothing
    end

    R1, R2, A1, A2 = halve_dest(R, A)
    nth2 = nth >> 1

    t = Threads.@spawn threaded_reducedim!(op, R1, A1, nth2)

    threaded_reducedim!(op, R2, A2, nth - nth2)

    wait(t)
    return nothing
end

function threaded_reducedim(op, A, dims, nth=Threads.nthreads())

    if nth == 1
        return serial_reducedim(op, A, dims)
    end

    A1, A2 = halve_last(A)
    nth2 = nth >> 1

    t = Threads.@spawn threaded_reducedim(op, A1, dims, nth2)

    op.(threaded_reducedim(op, A2, dims, nth - nth2), fetch(t))
end