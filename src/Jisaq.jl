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

include("JisaqCircuit.jl")
include("JisaqOperator.jl")
include("JisaqCircuitDrawer.jl")
include("JisaqStatevector.jl")
include("JisaqOpenMPS.jl")
include("JisaqBenchmark.jl")

end