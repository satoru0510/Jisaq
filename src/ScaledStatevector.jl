export ScaledStatevector, scaled_statevec

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

statevec(ssv::ScaledStatevector) = ssv |> vec |> statevec
scaled_statevec(sv::Statevector) = scaled_statevec(sv |> vec, 1)

Base.:*(sv::Statevector, s::Number) = scaled_statevec(sv.vec, s)
Base.:*(s::Number, sv::Statevector) = scaled_statevec(sv.vec, s)

function apply_diagonal1q!(ssv::ScaledStatevector, loc::Int, a,b)
    nq,vec = ssv.nq, ssv.vec
    bdiva = b / a
    @inbounds for i in 1 : 2^loc : 2^nq
        for j in i:i+2^(loc-1)-1
            vec[j] *= bdiva
        end
    end
    ssv.scalar *= a
    return ssv
end

function apply!(sv::ScaledStatevector, x::Rzz)
    nq,i,j = sv.nq, x.locs[1], x.locs[2]
    _i, _j = minmax(i,j)
    v = sv.vec
    theta = x.theta
    a,b = cis(theta/2), cis(-theta/2)
    mask1 = 2^(_i-1)
    mask2 = 2^(_j-1)
    bdiva = b / a
    for k in 0:2^(nq-2)-1
        idx1 = bit_insert(bit_insert(k, mask1), mask2)
        idx2 = idx1 | mask1 | mask2
        @inbounds v[idx1+1] *= bdiva
        @inbounds v[idx2+1] *= bdiva
    end
    sv.scalar *= a
    sv
end

function apply!(sv::ScaledStatevector, x::Scale)
    apply!(sv, x.op)
    sv.scalar *= x.scalar
end

function apply!(sv::ScaledStatevector, scalar::Number)
    sv.scalar *= scalar
    sv
end

inner_prod(sv1::AbstractStatevector, sv2::ScaledStatevector) = (vec(sv1) ⋅ sv2.vec) * sv2.scalar
inner_prod(sv1::ScaledStatevector, sv2::AbstractStatevector) = (sv1.vec ⋅ vec(sv2)) * sv1.scalar'
inner_prod(sv1::ScaledStatevector, sv2::ScaledStatevector) = (sv1.vec ⋅ sv2.vec) * sv1.scalar' * sv2.scalar

function expect(sv::ScaledStatevector, obs::AbstractChannel)
    cp = apply(sv, obs)
    (sv.vec ⋅ cp.vec) * sv.scalar' * cp.scalar
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

#TODO