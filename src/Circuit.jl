"""
    Circuit(gates::Vector{AbstractChannel})

A data type of quantum circuit consists of `gates`.
"""
struct Circuit{T<:AbstractChannel} <: AbstractChannel
    gates::AbstractVector{T}
    name::String
end

Circuit(gates) = Circuit(gates, "")

Base.:*(g1::AbstractChannel, g2::AbstractChannel) = Circuit([g2, g1])
Base.:*(g1::AbstractChannel, g2::AbstractChannel, g3::AbstractChannel) = Circuit([g3, g2, g1])
Base.:*(g1::AbstractChannel...) = Circuit(reverse!(vcat(g1...)))

Base.repeat(g::AbstractChannel, reps::Int) = Circuit(repeat([g], reps))

function Base.show(io::IO, cir::Circuit)
    println(io, "Circuit with $(length(cir.gates)) components")
    for i in cir.gates
        println(io, i)
    end
end

function apply!(sv::AbstractState, cir::Circuit; boundscheck=true)
    if boundscheck && nqubits(sv) < loc_max(cir)
        error("bounds error: loc_max=$(loc_max(cir)), nqubits(st)=$(nqubits(sv))")
    end
    for g in cir.gates
        apply!(sv, g)
    end
    sv
end

locs(cir::Circuit) = Tuple(1:loc_max(cir))
loc_max(cir::Circuit) = maximum(maximum.(locs.(cir.gates)))

function mat(nq::Int, cir::Circuit)
    mapreduce(mat(nq), *, cir.gates)
end