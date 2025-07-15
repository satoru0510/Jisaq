Jisaq - Simple and fast quantum circuit simulator.

# Installation
```julia
pkg> add https://github.com/satoru0510/Jisaq
```

# Basic Usage
## Statevector simulator

```julia
julia> using Jisaq

julia> sv = statevec(3) #3-qubit statevector initialized in zero state
2-qubit Statevector with Vector{ComplexF64}

julia> cir = Circuit([H(1), CX(1,2)])
Circuit with 2 components
H(1)

julia> draw(cir, :unicode)
─H─┬─
───X─

julia> result = apply(sv, cir) #inplace version apply! is also available
2-qubit Statevector with Vector{ComplexF64}

julia> expect(result, Z(1))
0.0 + 0.0im

julia> vec(result)
4-element Vector{ComplexF64}:
 0.7071067811865475 + 0.0im
                0.0 + 0.0im
                0.0 + 0.0im
 0.7071067811865475 + 0.0im
```

## Density matrix and MPS simulator
```julia
dmat = density(2)
2-qubit DensityMatrix with Matrix{ComplexF64}

mps = OpenMPS(8, 4)
8-qubit open boundary MPS (bond_max=4)
bond=[2, 4, 4, 4, 4, 4, 2]
```

# Supported gates
- Paulis: Id, X, Y, Z
- Basic single-qubit constant gates: H, S, T, Sdag, Tdag, U2
- Two-qubit controlled gates: CX, CZ
- Rotation gates: Rx, Ry, Rz, Rxx, Ryy, Rzz
- Efficient rotation gates: RyyRxx, RzzRyy, RxxRzz, etc.
- Projectors: P0, P1

For example, `RyyRxx(loc1, loc2, theta1, theta2)` is semantically equivalent to `Ryy(loc1, loc2, theta1) * Rxx(loc1, loc2, theta2)`, but more efficient.

# Benchmarks vs. [Yao.jl](https://github.com/QuantumBFS/Yao.jl)
<div align="center"> <img
src="https://satoru0510.github.io/assets/benchmark_jisaq_vs_yao.png"
alt="benchmark_plot"></img>
</div

- Trotterization of 1D periodic-boundary-condition transverse-field Heisenberg model's Hamiltonian (single step)
- Statevector Simulation
- OS: Linux (x86_64-linux-gnu)
- CPU: 8 × Intel(R) Xeon(R) W-2245 CPU @ 3.90GHz
- GPU: NVIDIA RTX A6000
