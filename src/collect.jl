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

function threaded_copyto!(dest, bc)

    n = length(dest)
    nth = Threads.nthreads()
    chunk = n ÷ nth 
    remainder = n % nth

    @inbounds Threads.@threads for t in 1:nth
        iter = ((t - 1) * chunk + 1):(t * chunk)
        @inbounds @simd for i in eachindex(bc)[iter]
            @inbounds dest[i] = bc[i]
        end
    end

    nlast = nth * chunk

    @inbounds @simd for i in eachindex(bc)[nlast:end]
        @inbounds dest[i] = bc[i]
    end


    nothing
end

# For GPUArrays dispatch to default which will call the corresponding GPU functions

@inline threaded_materialize(bc::Broadcast.Broadcasted{<:AbstractGPUArrayStyle}) = LazyTensor(Base.materialize(bc))

@inline threaded_materialize!(dest, bc::Broadcast.Broadcasted{<:AbstractGPUArrayStyle}) = Base.materialize!(bc)

# Fallback: non-lazy arrays pev and ev behaves as identity function

@inline ev(x) = x
@inline pev(x) = x