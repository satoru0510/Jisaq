export Operator, locs
using SparseArrays: sparse
using KrylovKit

"""
    Operator <: AbstractChannel

Abstract type of operator
"""
abstract type Operator <: AbstractChannel end

"""
    locs(::Operator) -> NTuple{N, Int}

returns the locations of qubits which the operator to be applied
"""

# define constant single-qubit unitary gates
for g in [:X, :Y, :Z, :H, :Id, :P0, :P1, :S, :T]
    name = string(g)
    @eval export $g
    @eval begin
        struct $g <: Operator
            loc::Int
        end

        Base.show(io::IO, x::$g) = print(io, string(nameof(typeof(x))), "($(x.loc))")
        locs(x::$g) = (x.loc,)

        @doc """
            $($name)(loc::Int)

        constant single-qubit operator `$($name)`
        """ $(g)
    end

end

export U2
"""
    U2(loc::Int, mat::T)

General single-qubit operator
"""
struct U2{T<:AbstractMatrix} <: Operator
    loc::Int
    mat::T
end
locs(x::U2) = (x.loc,)

export Controlled, controlled
struct Controlled{O<:Operator} <: Operator
    ctrl_loc::Int
    op::O
end

function controlled(ctrl_loc::Int, op::Operator)
    Controlled(ctrl_loc, op)
end

function Base.show(io::IO, x::Controlled)
    print(io, x.ctrl_loc, " controlled-", x.op )
end

locs(x::Controlled) = (x.ctrl_loc, locs(x.op)...)

export CX
"""
    CX(ctrl_loc::Int, targ_loc::Int)

CX (controlled-NOT) gate
"""
const CX = Controlled{X}

export CZ
"""
    CZ(loc1::Int, loc2::Int)

CZ (controlled-Z) gate
"""
const CZ = Controlled{Z}

# single-qubit rotation gate
for g in [:Rx, :Ry, :Rz]
    name = string(g)
    @eval export $g
    @eval begin
        struct $g <: Operator
            loc::Int
            theta::Float64
        end

        Base.show(io::IO, x::$g) = print(io, string(nameof(typeof(x))), "($(x.loc), θ=$(x.theta))")
        locs(x::$g) = (x.loc,)

        @doc """
            $($name)(loc::Int, theta::Float64)

        single-qubit rotation gate `$($name)`
        """ $(g)
    end
end

# multi-qubit pauli gate
for g in[:Xs, :Ys, :Zs]
    name = string(g)
    @eval export $g
    @eval begin
        struct $g{N} <: Operator
            locs::NTuple{N,Int}
        end

        function $g(locs::Int...)
            $g(locs)
        end

        function Base.show(io::IO, x::$g)
            print(io, string(nameof(typeof(x))), x.locs)
        end

        locs(x::$g) = x.locs

        @doc """
            $($name)(locs::Int...)

        multi-qubit pauli gate `$($name)`
        """ $(g)
    end
end

# two-qubit rotation gate
for g in [:Rxx, :Ryy, :Rzz]
    name = string(g)
    @eval export $g
    @eval begin
        struct $g <: Operator
            loc1::Int
            loc2::Int
            theta::Float64
        end

        Base.show(io::IO, x::$g) = print(io, string(nameof(typeof(x))), "($(x.loc1),$(x.loc2), θ=$(x.theta))")
        locs(x::$g) = (x.loc1, x.loc2)

        @doc """
            $($name)(loc1::Int, loc2::Int, theta::Float64)

        two-qubit rotation gate `$($name)`
        """ $(g)
    end
end

# two-qubit composit rotation gate
for g in [:RyyRxx, :RzzRyy, :RxxRzz, :RxxRyy, :RyyRzz, :RzzRxx]
    name = string(g)
    @eval export $g
    @eval begin
        struct $g <: Operator
            loc1::Int
            loc2::Int
            theta1::Float64
            theta2::Float64
        end

        Base.show(io::IO, x::$g) = print(io, string(nameof(typeof(x))), "($(x.loc1),$(x.loc2), θ1=$(x.theta1), θ2=$(x.theta2))")
        locs(x::$g) = (x.loc1, x.loc2)

        @doc """
            $($name)(loc1::Int, loc2::Int, theta1::Float64, theta2::Float64)

        two-qubit composit rotation gate `$($name)`
        """ $(g)
    end
end

"""
    I_plus_A{T,U}(loc1::Int, loc2::Int, d1::T, d2::T, b::U, c::U)

2-qubit unitary
    [
        d1 0 0 b
        0 d2 c 0
        0 c d2 0
        b 0 0 d1
    ]
"""
struct I_plus_A{T<:Number, U<:Number} <: Operator
    loc1::Int
    loc2::Int
    d1::T
    d2::T
    b::U
    c::U
end
locs(x::I_plus_A) = (x.loc1, x.loc2)
export I_plus_A

export mat
"""
    mat(::Type{<:Operator}, [θ1::Real, θ2::Real]) -> AbstractMatrix
    mat(nq::Int, op::T) where T<:Operator -> AbstractMatrix

returns matrix repesentation of a operator
"""
mat

Base.convert(::Type{Matrix}, x::Operator) = convert(Matrix, mat(x))
mat(::Type{X}) = sparse([0 1;1 0])
mat(::Type{Y}) = sparse([0 -im;im 0])
mat(::Type{Z}) = Diagonal([1, -1])
mat(::Type{Id}) = Diagonal([1, 1])
mat(::Type{H}) = [1 1;1 -1] / sqrt(2)
mat(::Type{P0}) = Diagonal([1, 0])
mat(::Type{P1}) = Diagonal([0, 1])
mat(::Type{S}) = Diagonal([1, im])
mat(::Type{T}) = Diagonal([1, exp(im*π/4)])
mat(::Type{CX}) = sparse([1 0 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0])
mat(::Type{CZ}) = Diagonal([1, 1, 1, -1])
mat(::Type{Rx}, θ::Real) = [cos(θ/2) -im*sin(θ/2); -im*sin(θ/2) cos(θ/2)]
mat(::Type{Ry}, θ::Real) = [cos(θ/2) -sin(θ/2); sin(θ/2) cos(θ/2)]
mat(::Type{Rz}, θ::Real) = Diagonal([cis(-θ/2), cis(θ/2)])
mat(x::Type{U2}) = x.mat
mat(::Type{Rxx}, θ::Real) = sparse([
    cos(θ/2) 0 0 -im*sin(θ/2)
    0 cos(θ/2) -im*sin(θ/2) 0
    0 -im*sin(θ/2) cos(θ/2) 0
    -im*sin(θ/2) 0 0 cos(θ/2)
])
mat(::Type{Ryy}, θ::Real) = sparse([
    cos(θ/2) 0 0 im*sin(θ/2)
    0 cos(θ/2) -im*sin(θ/2) 0
    0 -im*sin(θ/2) cos(θ/2) 0
    im*sin(θ/2) 0 0 cos(θ/2)
])
mat(::Type{Rzz}, θ::Real) = Diagonal([exp(-im*θ/2), exp(im*θ/2), exp(im*θ/2), exp(-im*θ/2)])
function mat(::Type{RyyRxx}, θ1::Real, θ2::Real)
    d1 = sin(θ1/2)*sin(θ2/2) + cos(θ1/2)*cos(θ2/2)
    d2 = -sin(θ1/2)*sin(θ2/2) + cos(θ1/2)*cos(θ2/2)
    b = im*(sin(θ1/2)*cos(θ2/2) - cos(θ1/2)*sin(θ2/2))
    c = im*(-sin(θ1/2)*cos(θ2/2) - cos(θ1/2)*sin(θ2/2))
    sparse([
        d1 0 0 b
        0 d2 c 0
        0 c d2 0
        b 0 0 d1
    ])
end
function mat(::Type{RzzRyy}, θ1::Real, θ2::Real)
    d1 = cos(θ1/2)*cos(θ2/2) - im*sin(θ1/2)*cos(θ2/2)
    d2 = cos(θ1/2)*cos(θ2/2) + im*sin(θ1/2)*cos(θ2/2)
    b = sin(θ1/2)*sin(θ2/2) + im*cos(θ1/2)*sin(θ2/2)
    c = sin(θ1/2)*sin(θ2/2) - im*cos(θ1/2)*sin(θ2/2)
    sparse([
        d1 0 0 b
        0 d2 c 0
        0 c d2 0
        b 0 0 d1
    ])
end
mat(::Type{I_plus_A}) = sparse([d1 0 0 b;0 d2 c 0;0 c d2 0;b 0 0 d1])

function mat(nq::Int, x::Union{X,Y,Z,H,Id,P0,P1,U2,S,T})
    m = mat(typeof(x)) |> sparse
    id = convert(typeof(m), I(2))
    loc = x.loc
    ret = loc==1 ? m : id
    for i in 2:nq
        if i==loc
            ret = kron(m, ret)
        else
            ret = kron(id, ret)
        end
    end
    ret
end

_basic_matrix(::Type{Xs{N}}) where N = mat(X)
_basic_matrix(::Type{Ys{N}}) where N = mat(Y)
_basic_matrix(::Type{Zs{N}}) where N = mat(Z)

function mat(nq::Int, x::Union{Xs, Ys, Zs})
    m = _basic_matrix(typeof(x)) |> sparse
    id = convert(typeof(m), I(2))
    locs = x.locs
    ret = 1 in locs ? m : id
    for i in 2:nq
        if i in locs
            ret = kron(m, ret)
        else
            ret = kron(id, ret)
        end
    end
    ret
end

function mat(nq::Int, x::Union{Rx,Ry,Rz})
    m = mat(typeof(x), x.theta) |> sparse
    id = convert(typeof(m), I(2))
    loc = x.loc
    ret = loc==1 ? m : id
    for i in 2:nq
        if i==loc
            ret = kron(m, ret)
        else
            ret = kron(id, ret)
        end
    end
    ret
end

Base.eltype(::Union{X,Z,Id,P0,P1,CX,CZ}) = Int
Base.eltype(::Union{H,Ry}) = Float64
Base.eltype(::Union{Rx,Rz,Rxx,Ryy,Rzz,RyyRxx,RzzRyy}) = ComplexF64
Base.eltype(x::U2) = eltype(x.mat)
Base.eltype(x::I_plus_A) = promote_type(typeof(x.d1), typeof(x.b))

export Add
"""
    Add{T<:AbstractVector{<:Operator}} <: Operator
    Add(contents::T)

represents sum of operators
"""
struct Add{T<:AbstractVector{<:Operator}} <: Operator
    contents::T
end

Base.:+(o1::Operator, o2::Operator) = Add([o1,o2])
Base.:+(o1::Add, o2::Operator) = Add(vcat(o1.contents, o2))
Base.:+(o1::Operator, o2::Add) = Add(vcat(o1, o2.contents))
Base.:+(o1::Add, o2::Add) = Add(vcat(o1.contents, o2.contents))

function Base.show(io::IO, x::Add)
    len = length(x.contents)
    for i in 1:len
        print(io, x.contents[i])
        if i != len
            print(io, " + ")
        end
    end
end


mat(nq::Int, x::Add) = sum(mat(nq, c) for c in x.contents)
locs(x::Add) = ([cc for c in x.contents for cc in locs(c)]...,)

mat(nq::Int) = (op::Operator) -> mat(nq, op)


export Scale
"""
    Scale{S<:Number, O<:Operator} <: Operator
    Scale(scalar::S, op::O)

represents a scaled operator
"""
struct Scale{S<:Number, O<:Operator} <: Operator
    scalar::S
    op::O
end
Base.:*(x::Number, s::Operator) = Scale(x, s)
Base.:\(x::Number, s::Operator) = Scale(inv(x), s)
Base.:*(x::Number, s::Scale) = Scale(x * s.scalar, s.op)
Base.:\(x::Number, s::Scale) = Scale(s.scalar / x, s.op)
Base.:*(s::Operator, x::Number) = Scale(x, s)
Base.:/(s::Operator, x::Number) = Scale(inv(x), s)
Base.:*(s::Scale, x::Number) = Scale(x * s.scalar, s.op)
Base.:/(s::Scale, x::Number) = Scale(s.scalar / x, s.op)
Base.:-(s::Operator) = Scale(-1, s)
Base.:-(s::Scale) = Scale(-1 * s.scalar, s.op)

Base.show(io::IO, x::Scale) = print(io, "Scale($(x.scalar), x.op)")

mat(nq::Int, x::Scale) = mat(nq, x.op) * x.scalar
locs(x::Scale) = locs(x.op)

export TimeEvolution
struct TimeEvolution{O<:Operator, T<:Number} <: Operator
    hamilt::O
    t::T
end
mat(nq::Int, te::TimeEvolution) = exp(Matrix(mat(nq, -im * te.hamilt * te.t)))
locs(te::TimeEvolution) = locs(te.hamilt)

#TODO