export Statevector, StatevectorSimulator
export statevec, rand_statevec, expect, nqubits

using LinearAlgebra, Random

"""
    mutable struct Statevector{T <: AbstractVector} <: AbstractState
"""
mutable struct Statevector{T <: AbstractVector} <: AbstractState
    nq::Int
    vec::T
    function Statevector(vec::AbstractVector)
        nq = length(vec) |> log2 |> Int
        new{typeof(vec)}(nq, vec)
    end
end

"""
    Statevector([ty=ComplexF64::Type,] nq::Int)

Returns `nq`-qubit zero state with type `ty`
"""
Statevector(ty::Type, nq::Int) = statevec(ty, nq)
Statevector(nq::Int) = statevec(nq)

"""
    statevec([ty::Type{T}=ComplexF64], nq::Int)

Returns `nq`-qubit zero state with type `ty`
"""
function statevec(ty::Type, nq::Int)
    ret = Statevector(zeros(ty, 2^nq))
    ret.vec[1] = 1
    ret
end

function statevec(nq::Int)
    statevec(ComplexF64, nq)
end

"""
    statevec(vec::AbstractVector) -> Statevector

Wraps `vec` as a `Statevector`.
"""
function statevec(vec::AbstractVector)
    Statevector(vec)
end

function Base.show(io::IO, sv::Statevector)
    nq = sv.nq
    print(io, "$nq-qubit Statevector with $(typeof(sv.vec))")
end

"""
    vec(sv::Statevector)

Return statevector of `sv`.
"""
Base.vec(sv::Statevector) = sv.vec

"""
    nqubits(sv::Statevector)

Returns number of qubits of `sv`.
"""
nqubits(sv::Statevector) = sv.nq

"""
    copy(sv::Statevector)

Make a copy of a Statevector `sv`. It is equivalent to `Statevector(copy(sv.vec))`.
"""
Base.copy(sv::Statevector) = Statevector(copy(sv.vec))


# helper functions
@inline function bit_insert(a, _2_pow_idx)
    rem = a % _2_pow_idx
    (a ⊻ rem) << 1 + rem
end

@inline function swap_rows!(B::AbstractVector, i::Integer, j::Integer)
    @inbounds B[i], B[j] = B[j], B[i]
    B
end

@inline function swap_rows!(v::AbstractVector, i::AbstractVector{T}, j::AbstractVector{T}) where T<:Integer
    for k in 1:length(i)
        @inbounds swap_rows!(v, i[k], j[k])
    end
    v
end

apply!(sv::Statevector, ::Id) = sv

function apply!(sv::Statevector, x::X)
    nq,loc,vec = sv.nq, x.loc, sv.vec
    _2_pow_locm1 = 2^(loc - 1)
    _2_pow_loc_m1 = 2^loc-1
    for i in 1 : 2^loc : 2^nq
        swap_rows!(vec, i : i+_2_pow_locm1-1, i+_2_pow_locm1 : i+_2_pow_loc_m1)
    end
    return sv
end

function apply!(sv::Statevector, x::Y)
    nq,loc,vec = sv.nq, x.loc, sv.vec
    _2_pow_locm1 = 2^(loc - 1)
    _2_pow_loc_m1 = 2^loc-1
    for i in 1 : 2^loc : 2^nq
        swap_rows!(vec, i : i+_2_pow_locm1-1, i+_2_pow_locm1:i+_2_pow_loc_m1)
        for j in i : i+_2_pow_locm1-1
            @inbounds vec[j] *= -im
        end
        for j in i+_2_pow_locm1 : i+_2_pow_loc_m1
            @inbounds vec[j] *= im
        end
    end
    return sv
end

function apply!(sv::Statevector, x::Z)
    nq,loc,vec = sv.nq, x.loc, sv.vec
    for i in 1 : 2^loc : 2^nq
        for j in i+2^(loc-1):i+2^loc-1
            @inbounds vec[j] *= -1
        end
    end
    return sv
end

function apply!(sv::Statevector, x::U2)
    nq,loc,vec = sv.nq, x.loc, sv.vec
    a,c,b,d = x.mat
    step1 = 1 << (loc - 1)
    step2 = 1 << loc
    for j in 1 : step2 : size(vec, 1)-step1+1
        @inbounds for i in j : j+step1-1
            w = vec[i]
            v = vec[i+step1]
            vec[i] = a * w + b * v
            vec[i+step1] = c * w + d * v
        end
    end
    return sv
end

const _hadamard_mat = [1 1; 1 -1] / sqrt(2)
function apply!(sv::Statevector, x::H)
    apply!(sv, U2(x.loc, _hadamard_mat))
end

function apply!(sv::Statevector, x::CX)
    nq,v = sv.nq, sv.vec
    i,j = x.ctrl_loc, x.targ_loc
    offset = 1 + 2^(i-1)
    step = 2^(j-1)
    _i, _j = minmax(i,j)
    _2_pow_i = 1 << (_i-1)
    _2_pow_j = 1 << (_j-1)
    for k in 0 : 2^(nq-2)-1
        first = bit_insert(bit_insert(k, _2_pow_i), _2_pow_j) + offset
        swap_rows!(v, first, first+step)
    end
    sv
end

function apply!(sv::Statevector, x::Rx)
    theta = x.theta/2
    m = [
        cos(theta) -im*sin(theta)
        -im*sin(theta) cos(theta)
        ]
    apply!(sv, U2(x.loc, m))
end

function apply!(sv::Statevector, x::Ry)
    theta = x.theta/2
    m = [
        cos(theta) -sin(theta)
        sin(theta) cos(theta)
        ]
    apply!(sv, U2(x.loc, m))
end

function apply_diagonal1q!(sv::Statevector, a,b)
        nq,loc,vec = sv.nq, x.loc, sv.vec
        @inbounds for i in 1 : 2^loc : 2^nq
            for j in i:i+2^(loc-1)-1
                vec[j] *= b
            end
            for j in i+2^(loc-1):i+2^loc-1
                vec[j] *= a
            end
        end
        return sv
end

function apply!(sv::Statevector, x::Rz)
    a,b = cis(x.theta/ 2), cis(-x.theta/ 2)
    apply_diagonal1q!(sv, a,b)
end

function apply!(sv::Statevector, x::Xs)
    nq = sv.nq
    idxs = x.locs
    v = sv.vec
    maxi = maximum(idxs)-1
    _2_pow_maxi = 2^maxi
    mask = sum(2^(i-1) for i in idxs)
    for i in 0 : 2^(nq-1)-1
        l1 = bit_insert(i, _2_pow_maxi)
        swap_rows!(v, l1+1, l1⊻mask+1)
    end
    sv
end

function apply!(sv::Statevector, x::I_plus_A)
    a1,a2,b,c = x.d1, x.d2, x.b, x.c
    i,j,nq,v = x.loc1, x.loc2, sv.nq, sv.vec
    mask = 2^(i-1) + 2^(j-1)
    mini,maxi = minmax(i,j)
    _2_pow_maxim1 = 2^(maxi-1)
    step1 = 2^mini
    step2 = 2^(mini-1)
    for l in 0 : step1 : 2^nq-1
        selector = l & _2_pow_maxim1 == 0
        bc = selector ? b : c
        a12 = selector ? a1 : a2
        @inbounds for k in 0:step2-1
            idx1 = l+k+1
            idx2 = (l+k)⊻mask+1
            x = v[idx1]
            y = v[idx2]
            v[idx1] = x * a12 + y * bc
            v[idx2] = x * bc + y * a12
        end
    end
end

function apply!(sv::Statevector, x::Rxx)
    theta = x.theta/2
    d,bc = cos(theta), -im*sin(theta)
    apply!(sv, I_plus_A(x.loc1, x.loc2, d, d, bc, bc))
end

function apply!(sv::Statevector, x::Ryy)
    theta = x.theta/2
    s = sin(theta)
    d,b,c = cos(theta), im*s, -im*s
    apply!(sv, I_plus_A(x.loc1, x.loc2, d,d,b,c))
end

function apply!(sv::Statevector, x::RyyRxx)
    yy,xx = x.theta1/2, x.theta2/2
    sx,sy,cx,cy = sin(xx), sin(yy), cos(xx), cos(yy)
    d1 = sx * sy + cy * cx
    d2 = -sx * sy + cy * cx
    b = im * (-sx * cy + sy * cx)
    c = im * (-sx * cy - sy * cx)
    apply!(sv, I_plus_A(x.loc1, x.loc2, d1,d2,b,c))
end

function apply!(sv::Statevector, x::RzzRyy)
    zz,yy = x.theta1/2, x.theta2/2
    sy,sz,cy,cz = sin(yy), sin(zz), cos(yy), cos(zz)
    d1 = cz * cy - im * sz * cy
    d2 = cy * cz + im * sz * cy
    b = sy * sz + im * cz * sy
    c = sy * sz - im * sy * cz
    apply!(sv, I_plus_A(x.loc1, x.loc2, d1,d2,b,c))
end

function apply!(sv::Statevector, x::Rzz)
    nq,i,j = sv.nq, x.loc1, x.loc2
    v = sv.vec
    theta = x.theta
    a,b = cis(theta/2), cis(-theta/2)
    mask1 = 2^(i-1)
    mask2 = 2^(j-1)
    for k in 0:2^nq-1
        bit1 = k & mask1 != 0
        bit2 = k & mask2 != 0
        @inbounds v[k+1] *= (bit1 != bit2) ? a : b
    end
end

function apply!(sv::Statevector, te::TimeEvolution)
    h = te.hamilt
    nq = nqubits(sv)
    h_mat = mat(nq, h)
    v,_ = exponentiate(-im*h_mat, te.t, sv.vec)
    sv.vec = v
end

"""
    expect(sv::Statevector, obs::AbstractChannel)

compute expectation value
"""
function expect(sv::Statevector, obs::AbstractChannel)
    cp = apply(sv, obs)
    sv.vec ⋅ cp.vec
end

function rand_statevec(rng, nq::Int)
    raw = randn(rng, ComplexF64, 2^nq) |> statevec
    normalize!(raw)
end

function rand_statevec(nq::Int)
    rand_statevec(Random.default_rng(), nq)
end

"""
    rand_statevec([rng,] nq::Int) -> Statevector

Returns random `nq`-qubit statevector.
"""
rand_statevec

for fun in [:norm, :normalize, :normalize!]
    @eval export $fun
    @eval function LinearAlgebra.$fun(sv::Statevector)
        $fun(sv.vec)
        sv
    end
end

mutable struct StatevectorSimulator <: Simulator
    boundscheck::Bool
    callback::Function
end

"""
    StatevectorSimulator(;boundscheck::Bool=true, callback::Function=_->false)

Statevector quantum circuit simulator.
- `boundscheck`: wether check the bounds of statevector and operator
- `callback`: a  function(`Statevector` -> `Bool`) called at end of each circuit element application. If it return true, simulation is aborted.
"""
StatevectorSimulator(;boundscheck::Bool=true, callback::Function=_->false) = StatevectorSimulator(boundscheck, callback)

export run, run!
"""
    run!(init::Statevector, cir::Circuit, sim::StatevectorSimulator)

inplace version of `run(init::Statevector, cir::Circuit, sim::StatevectorSimulator)`
"""
function run!(init::Statevector, cir::Circuit, sim::StatevectorSimulator)
    nq = nqubits(init)
    if sim.boundscheck
        for g in cir.gates
            if !prod(i in 1:nq for i in locs(g))
                error("bound error: $nq-qubit Statevector but locs($(g))=$(locs(g))")
            end
        end
    end

    for g in cir.gates
        apply!(init, g)
        if sim.callback(init)
            println(stderr, "circuit simulation was aborted")
            return init
        end
    end
    init
end

"""
    run(init::Statevector, cir::Circuit, sim::StatevectorSimulator)

run circuit simulation
"""
Base.run(init::Statevector, cir::Circuit, sim::StatevectorSimulator) = run!(copy(init), cir, sim)
Base.run(init::Statevector, cir::Circuit) = run(init, cir, StatevectorSimulator())