export OpenMPS
export nqubits, apply!, expect
using LinearAlgebra, TensorKit

"""
    OpenMPS <: AbstractState

open boundary conditioned MPS.
- `tens::Vector{TensorMap{ComplexF64, ComplexSpace, 1, 2, Vector{ComplexF64}}}`
tensors of the MPS. `length(tens)` is number of qubits

- `bond_max::Int`
maximal bond dimension
"""
struct OpenMPS <: AbstractState
    tens::Vector{TensorMap{ComplexF64, ComplexSpace, 1, 2, Vector{ComplexF64}}}
    bond_max::Int
end

nqubits(x::OpenMPS) = length(x.tens)

function Base.show(io::IO, mps::OpenMPS)
    nq = nqubits(mps)
    println(io, nq, "-qubit open boundary MPS (bond_max=$(mps.bond_max))")
    print(io, "bond=", [domain(mps.tens[i])[2].d for i in 1:nq-1])
end

function OpenMPS(nq::Int, bond_max::Int)
    tens = Vector{TensorMap{ComplexF64, ComplexSpace, 1, 2, Vector{ComplexF64}}}(undef, nq)
    for i in 1:nq÷2
        left = min(2^(i-1), bond_max)
        right = min(2^i, bond_max)
        tens[i] = zeros(ComplexF64, ℂ^2, ℂ^left ⊗ (ℂ^right)')
        tens[nq-i+1] = zeros(ComplexF64, ℂ^2, ℂ^right ⊗ (ℂ^left)')
        tens[i][1] = tens[nq-i+1][1] = 1
    end
    if isodd(nq)
        bond = min(2^(nq÷2), bond_max)
        tens[nq÷2+1] = zeros(ComplexF64, ℂ^2, ℂ^bond ⊗ (ℂ^bond)')
        tens[nq÷2+1][1] = 1
    end
    OpenMPS(tens, bond_max)
end

function contract(x::TensorMap{ComplexF64, ComplexSpace, 1, 2, Vector{ComplexF64}}, y::TensorMap{ComplexF64, ComplexSpace, 1, 2, Vector{ComplexF64}})
    @tensor ret[i1,i2;i3,i4] := x[i1,i3,a] * y[i2,a,i4]
end

function svd_mps(x::TensorMap{ComplexF64, ComplexSpace, 2, 2, Vector{ComplexF64}}; kwargs...)
    U,S,V,e = tsvd(x, (1,3), (2,4); kwargs...)
    B1 = permute(U, (1,), (2,3))
    B2 = permute(S*V, (2,), (1,3))
    # (B1, B2)
    a = domain(x)[1]
    b = domain(x)[2]
    c = domain(S)
    TensorMap(B1.data, ℂ^2 ← (a ⊗ (c)')), TensorMap(B2.data, ℂ^2 ← (c ⊗ b))
end

function contract_all(x1::TensorMap, x2::TensorMap)
    l1 = domain(x1).spaces[1].d
    sz1 = codomain(x1).spaces[1].d
    l2 = domain(x2).spaces[2].d
    sz2 = codomain(x2).spaces[1].d
    @tensor ret[i1,i2,i3,i4] := x1[i1,i3,a] * x2[i2,a,i4]
    TensorMap(ret.data, ℂ^(sz1*sz2), ℂ^l1 ⊗ (ℂ^l2)')
end

function apply!(mps::OpenMPS, gate::TensorMap{T, ComplexSpace, 1, 1, Vector{T}}, loc::Int) where T<:Number
    mps.tens[loc] = gate * mps.tens[loc]
    mps
end

function apply!(mps::OpenMPS, gate::TensorMap{T, ComplexSpace, 2, 2, Vector{T}}, loc::Tuple{Int, Int}) where T<:Number
    if abs(loc[1] - loc[2]) != 1
        error("non-local gate application is not supported")
    end
    loc1, loc2 = minmax(loc[1], loc[2])
    temp = contract(mps.tens[loc1], mps.tens[loc2])
    temp = gate * temp
    mps.tens[loc1], mps.tens[loc2] = svd_mps(temp; trunc=truncdim(mps.bond_max))
    mps
end

function apply!(mps::OpenMPS, gate::TensorMap{T, ComplexSpace, 2, 2, Vector{T}}, loc1::Int, loc2::Int) where T<:Number
    apply!(mps, gate, (loc1, loc2))
end

function to_vec(mps::OpenMPS)
    to_vec(mps.tens...).data
end

function to_vec(tens::TensorMap{ComplexF64, ComplexSpace, 1, 2, Vector{ComplexF64}}...)
    to_vec(tens[1], to_vec(tens[2:end]...))
end

function to_vec(
        ten1::TensorMap{ComplexF64, ComplexSpace, 1, 2, Vector{ComplexF64}}, 
        ten2::TensorMap{ComplexF64, ComplexSpace, 1, 2, Vector{ComplexF64}}
    )
    contract_all(ten1, ten2)
end

"""
    statevec(mps::OpenMPS)

convert MPS `mps` to Statevector
"""
function statevec(mps::OpenMPS)
    statevec(to_vec(mps))
end

function inner_prod(m1::OpenMPS, m2::OpenMPS)
    @assert nqubits(m1) == nqubits(m2)
    nq = nqubits(m1)
    @tensor acc[i1,i2,i3,i4] := conj(m1.tens[1][a,i1,i2]) * m2.tens[1][a,i3,i4]
    for i in 2:nq
        @tensor tmp[i1,i2,i3,i4] := conj(m1.tens[i][a,i1,i2]) * m2.tens[i][a,i3,i4]
        @tensor acc[i1,i2,i3,i4] := acc[i1,a,i3,b] * tmp[a,i2,b,i4]
    end
    acc[1]
end

function Base.copy(mps::OpenMPS)
    OpenMPS(copy(mps.tens), mps.bond_max)
end

function expect(mps::OpenMPS, obs::Pair{Int, TensorMap{T, ComplexSpace, 1, 1, Vector{T}}}...) where T<:Number
    cp = copy(mps)
    for o in obs
        apply!(cp, o[2], o[1])
    end
    inner_prod(mps, cp)
end

"""
    expect(mps::OpenMPS, obs::AbstractChannel)

compute expectation value
"""
function expect(mps::OpenMPS, obs::AbstractChannel)
    cp = apply(mps, obs)
    inner_prod(mps, cp)
end

apply!(mps::OpenMPS, x::Id) = mps

function apply!(mps::OpenMPS, x::Union{X,Y,Z,H,U2,P0,P1,T,S,Tdag,Sdag})
    m = mat(typeof(x))
    t = TensorMap(m, ℂ^2, ℂ^2)
    apply!(mps, t, x.locs[1])
end

function apply!(mps::OpenMPS, x::Union{Rx,Ry,Rz})
    m = mat(typeof(x), x.theta)
    t = TensorMap(m, ℂ^2, ℂ^2)
    apply!(mps, t, x.locs[1])
end

function apply!(mps::OpenMPS, x::Union{CX,CZ})
    m = mat(typeof(x))
    t = TensorMap(m, ℂ^2 * ℂ^2, ℂ^2 * ℂ^2)
    apply!(mps, t, x.locs...)
end

function apply!(mps::OpenMPS, x::Union{Rxx, Ryy, Rzz})
    m = mat(typeof(x), x.theta)
    t = TensorMap(m, ℂ^2 * ℂ^2, ℂ^2 * ℂ^2)
    apply!(mps, t, x.locs...)
end

function apply!(mps::OpenMPS, x::Union{RyyRxx, RzzRyy, RxxRzz, RxxRyy, RyyRzz, RzzRxx})
    m = mat(typeof(x), x.theta1, x.theta2)
    t = TensorMap(m, ℂ^2 * ℂ^2, ℂ^2 * ℂ^2)
    apply!(mps, t, x.locs...)
end