export ScaledStatevector

mutable struct ScaledStatevector{T <: AbstractVector} <: AbstractStatevector
    nq::Int
    vec::T
    scalar::ComplexF64
    
    function ScaledStatevector(vec::AbstractVector, scalar::ComplexF64)
        nq = length(vec) |> log2 |> Int
        new{typeof(vec)}(nq, vec, scalar)
    end
end

"""
    ScaledStatevector([ty=ComplexF64::Type,] nq::Int)

Returns `nq`-qubit zero state `ScaledStatevector` with type `ty`
"""
ScaledStatevector(ty::Type, nq::Int) = scaled_statevec(ty, nq)
ScaledStatevector(nq::Int) = scaled_statevec(nq)

"""
    scaled_statevec([ty::Type{T}=ComplexF64], nq::Int, [scalar::Number=1])

Returns `nq`-qubit zero state `ScaledStatevector` with type `ty`
"""
function scaled_statevec(ty::Type, nq::Int, scalar::Number)
    ret = ScaledStatevector(zeros(ty, 2^nq), convert(ComplexF64, scalar) )
    ret.vec[1] = 1
    ret
end

function scaled_statevec(ty::Type, nq::Int)
    ret = ScaledStatevector(zeros(ty, 2^nq), one(ComplexF64) )
    ret.vec[1] = 1
    ret
end

function scaled_statevec(nq::Int)
    scaled_statevec(ComplexF64, nq)
end

"""
    scaled_statevec(vec::AbstractVector, scalar::Number) -> ScaledStatevector

Wraps `vec` as a `ScaledStatevector`.
"""
function scaled_statevec(vec::AbstractVector, scalar::Number)
    ScaledStatevector(vec, convert(ComplexF64, scalar) )
end

function scaled_statevec(nq::Int, scalar::Number)
    scaled_statevec(ComplexF64, nq, scalar)
end

function Base.show(io::IO, sv::ScaledStatevector)
    nq = sv.nq
    print(io, "$nq-qubit ScaledStatevector with $(typeof(sv.vec)) scalar=$(sv.scalar)")
end

"""
    vec(sv::ScaledStatevector)

Return statevector of `sv`.
"""
Base.vec(sv::ScaledStatevector) = sv.vec * sv.scalar

"""
    copy(sv::ScaledStatevector)

Make a copy of a ScaledStatevector `sv`.
"""
Base.copy(sv::ScaledStatevector) = ScaledStatevector(copy(sv.vec), sv.scalar)

#TODO
#Statevector <-> ScaledStatevector
#apply_diagonal1q!

function apply!(sv::ScaledStatevector, x::Rzz)
end

function expect(sv::ScaledStatevector, obs::AbstractChannel)
end

LinearAlgebra.norm(sv::ScaledStatevector) = norm(sv.vec) * abs(sv.scalar)^2

function Base.:*(ssv::ScaledStatevector, s::Number)
    ssv.scalar = ssv.scalar * s
    ssv
end

function Base.:*(s::Number, ssv::ScaledStatevector)
    ssv.scalar = s * ssv.scalar
    ssv
end