module Jisaq

export AbstractState, AbstractChannel, Simulator, Circuit
export apply!, apply
abstract type AbstractState end
abstract type AbstractChannel end

"""
abstract type Simulator
"""
abstract type Simulator end

"""
    apply!(state::AbstractState, ch::AbstractChannel)

Apply a channel `ch` to `state` inplace
"""
apply!

"""
    apply(st::AbstractState, cir::AbstractChannel)

Apply a channel `cir` to state `st`. `st` will not changed.
"""
function apply(st::AbstractState, cir::AbstractChannel)
    ret = copy(st)
    apply!(ret, cir)
    ret
end

function Base.:*(cir::AbstractChannel, st::AbstractState)
    apply(st, cir)
end

function (cir::AbstractChannel)(st::AbstractState)
    apply(st, cir)
end

export cu_statevec
"""
    cu_statevec([ty::Type{T}=ComplexF64], nq::Int)

CUDA version of `statevec`
"""
function cu_statevec end

export cu_rand_statevec
"""
    cu_rand_statevec([ty::Type{T}=ComplexF64], nq::Int)

CUDA version of `rand_statevec`
"""
function cu_rand_statevec end

include("Circuit.jl")
include("Operator.jl")
include("CircuitDrawer.jl")
include("Statevector.jl")
include("ScaledStatevector.jl")
include("OpenMPS.jl")
include("Benchmark.jl")

end