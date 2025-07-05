export AbstractDensityMatrix, DensityMatrix, density

struct Yconj <: Operator
    locs::Tuple{Int}
end

#conjugate-invariant
for gt in [:Id,:X,:Z,:H,:P0,:P1,:Ry]
    @eval Main.conj(x::$gt) = x
end

#conjugate-antivariant
for gt in [:Rx,:Rz]
    @eval Main.conj(x::$gt) = $gt(x.locs, -x.theta)
end

Main.conj(x::U2) = U2(x.locs, conj(x.mat) )
Main.conj(x::S) = Sdag(x.locs)
Main.conj(x::Sdag) = S(x.locs)
Main.conj(x::T) = Tdag(x.locs)
Main.conj(x::Tdag) = T(x.locs)
Main.conj(x::Y) = Yconj(x.locs)

function apply!(sv::AbstractStatevector, x::Yconj)
    nq,loc,vec = sv.nq, x.locs[1], sv.vec
    _2_pow_locm1 = 2^(loc - 1)
    _2_pow_loc_m1 = 2^loc-1
    for i in 1 : 2^loc : 2^nq
        swap_rows!(vec, i : i+_2_pow_locm1-1, i+_2_pow_locm1:i+_2_pow_loc_m1)
        for j in i : i+_2_pow_locm1-1
            @inbounds vec[j] *= im
        end
        for j in i+_2_pow_locm1 : i+_2_pow_loc_m1
            @inbounds vec[j] *= -im
        end
    end
    return sv
end

Main.conj(x::I_plus_A) = I_plus_A(x.locs, x.d1', x.d2', x.b', x.c')
Main.conj(x::CX) = x
Main.conj(x::CZ) = x

for g in [:Rxx, :Ryy, :Rzz]
    @eval Main.conj(x::$g) = $g(x.locs, -x.theta)
end

for g in [:RxxRyy, :RyyRzz, :RzzRxx, :RzzRyy, :RyyRxx, :RxxRzz]
    @eval Main.conj(x::$g) = $g(x.locs, -x.theta1, -x.theta2)
end

abstract type AbstractDensityMatrix <: AbstractState end

mutable struct DensityMatrix{T<:AbstractMatrix} <: AbstractDensityMatrix
    mat::T
end

function density(ty::Type{<:Number}, nq::Int)
    m = zeros(ty, 2^nq, 2^nq)
    m[1] = 1
    DensityMatrix(m)
end

function density(sv::Statevector)
    DensityMatrix(sv.vec * sv.vec')
end

function Base.show(io::IO, dm::DensityMatrix)
    nq = nqubits(dm)
    print(io, "$nq-qubit DensityMatrix with $(typeof(dm.mat))")
end

density(nq::Int) = density(ComplexF64, nq)

nqubits(dm::DensityMatrix) = log2(size(dm.mat)[1] ) |> Int

function apply!(dm::DensityMatrix, x::Operator)
    nq = nqubits(dm)
    vdm = Statevector(reshape(dm.mat, 4^nq) )
    apply!(vdm, x)
    xx = deepcopy(x)
    xx.locs = xx.locs .+ nq
    apply!(vdm, conj(xx) )
    dm
end

function apply!(dm::DensityMatrix, x::Controlled)
    nq = nqubits(dm)
    vdm = Statevector(reshape(dm.mat, 4^nq) )
    apply!(vdm, x)
    xx = deepcopy(x)
    xx.ctrl_loc += nq
    xx.op.locs = xx.op.locs .+ nq
    apply!(vdm, conj(xx) )
    dm
end

function expect(dm::DensityMatrix, op::Union{Operator, Circuit{<:Operator}})
    nq = nqubits(dm)
    tr(dm.mat * mat(nq, op) )
end

function Base.copy(dmat::Jisaq.DensityMatrix)
    Jisaq.DensityMatrix(copy(dmat.mat) )
end

mat(dm::DensityMatrix) = dm.mat