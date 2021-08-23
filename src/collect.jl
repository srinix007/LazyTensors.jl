using GPUArrays

@inline vect(bc::Broadcast.Broadcasted{LazyTensorStyle}) = pev(bc)

"""
    ev(bc::Broadcast.Broadcasted{LazyTensorStyle}) 

Evaluates the Broadcasted object serially.

"""
@inline ev(bc::Broadcast.Broadcasted{LazyTensorStyle}) = LazyTensor(Base.materialize(unwrap(bc)))

"""
    pev(bc::Broadcast.Broadcasted{LazyTensorStyle}) 

Evaluates the Broadcasted object parallely in a device
agnostic way. 

"""
@inline pev(bc::Broadcast.Broadcasted{LazyTensorStyle}) = LazyTensor(threaded_materialize(unwrap(bc)))

function threaded_materialize(bc::Broadcast.Broadcasted{<:Broadcast.AbstractArrayStyle})
    bci = Broadcast.instantiate(bc)
    Eltype = Broadcast.combine_eltypes(bci.f, bci.args)
    dest = similar(bci, Eltype)
    axes(dest) == axes(bci) || Broadcast.throwdm(axes(dest), axes(bci))

    bcp = Broadcast.preprocess(dest, bci)

    threaded_copyto!(dest, bci)

    return LazyTensor(dest)
end

@inline function Base.materialize!(dest, bc::Broadcast.Broadcasted{LazyTensorStyle})
    threaded_copyto!(dest.dat, Broadcast.instantiate(unwrap(bc)))
    dest
end

function threaded_copyto!(dest, bc, st=1, en=length(bc), nth=Threads.nthreads())

    if nth == 1
        serial_copyto!(dest, bc, st, en)
        return nothing
    end

    mid = (st + en) >>> 1
    nth2 = nth >>> 1

    t = Threads.@spawn threaded_copyto!(dest, bc, st, mid, nth2)

    threaded_copyto!(dest, bc, mid + 1, en, nth - nth2)

    wait(t)

    return nothing

end

function serial_copyto!(dest, bc, st, en)

    @inbounds @simd for i in view(eachindex(bc), st:en)
        @inbounds dest[i] = bc[i]
    end

end

# For GPUArrays dispatch to default which will call the corresponding GPU functions

@inline threaded_materialize(bc::Broadcast.Broadcasted{<:AbstractGPUArrayStyle}) = LazyTensor(Base.materialize(bc))

@inline threaded_materialize!(dest, bc::Broadcast.Broadcasted{<:AbstractGPUArrayStyle}) = Base.materialize!(bc)

# Fallback: non-lazy arrays pev and ev behaves as identity function

@inline ev(x) = x
@inline pev(x) = x