"""
    Circuit(gates::Vector{AbstractChannel})

A data type of quantum circuit consists of `gates`.
"""
struct Circuit{T<:AbstractChannel} <: AbstractChannel
    gates::AbstractVector{T}
    name::String
end

Circuit(gates) = Circuit(gates, "")

Base.:*(g1::AbstractChannel, g2::AbstractChannel) = Circuit([g1, g2])
Base.:*(g1::AbstractChannel...) = Circuit(vcat(g1...))

Base.repeat(g::AbstractChannel, reps::Int) = Circuit(repeat([g], reps))

function Base.show(io::IO, cir::Circuit)
    println(io, "Circuit with $(length(cir.gates)) components")
    for i in cir.gates
        println(io, i)
    end
end

function apply!(sv::AbstractState, cir::Circuit)
    #TODO: boundscheck
    for g in cir.gates
        apply!(sv, g)
    end
    sv
end

locs(cir::Circuit) = Tuple(1:loc_max(cir))