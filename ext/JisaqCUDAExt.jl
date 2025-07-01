module JisaqCUDAExt

using Jisaq, CUDA

CUDA.allowscalar(false)

function Jisaq.cu_statevec(T, nq::Int)
    ret = CUDA.zeros(T, 2^nq)
    function k(a)
        a[1] = 1
        return
    end
    @cuda k(ret)
    Statevector(ret)
end

function Jisaq.cu_statevec(nq::Int)
    cu_statevec(ComplexF64, nq)
end

function bit_insert(a, _2_pow_idx)
    rem = a % _2_pow_idx
    (a ⊻ rem) << 1 + rem
end

function Jisaq.apply!(cusv::Statevector{<:CuArray}, x::X)
    function k(a, loc)
        i = threadIdx().x - 1
        j = blockIdx().x - 1
        idx = 1024j + i
        lm1 = loc - 1
        idx1 = bit_insert(idx, 1 << lm1 )
        idx2 = idx1 ⊻ 1 << lm1
        idx1 += 1
        idx2 += 1
        a[idx1],a[idx2] = a[idx2], a[idx1]
        return
    end
    loc = x.loc
    arr = cusv.vec
    if length(arr) ≤ 1024
        @cuda blocks=1 threads=length(arr)÷2 k(arr, loc)
    else
        @cuda blocks=length(arr)÷(2048) threads=1024 k(arr, loc)
    end
    cusv
end

function Jisaq.apply!(cusv::Statevector{<:CuArray}, x::Y)
    function k(a, loc)
        i = threadIdx().x - 1
        j = blockIdx().x - 1
        idx = 1024j + i
        lm1 = loc - 1
        idx1 = bit_insert(idx, 1 << lm1 )
        idx2 = idx1 ⊻ 1 << lm1
        idx1 += 1
        idx2 += 1
        a[idx1],a[idx2] = a[idx2]*(-im), a[idx1]*im
        return
    end
    loc = x.loc
    arr = cusv.vec
    if length(arr) ≤ 1024
        @cuda blocks=1 threads=length(arr)÷2 k(arr, loc)
    else
        @cuda blocks=length(arr)÷(2048) threads=1024 k(arr, loc)
    end
    cusv
end

function Jisaq.apply!(cusv::Statevector{<:CuArray}, x::Z)
    function k(a, loc)
        i = threadIdx().x - 1
        j = blockIdx().x - 1
        idx = 1024j + i
        lm1 = loc - 1
        idx1 = bit_insert(idx, 1 << lm1 )
        idx2 = idx1 ⊻ 1 << lm1 + 1
        a[idx2] *= -1
        return
    end
    loc = x.loc
    arr = cusv.vec
    if length(arr) ≤ 1024
        @cuda blocks=1 threads=length(arr)÷2 k(arr, loc)
    else
        @cuda blocks=length(arr)÷(2048) threads=1024 k(arr, loc)
    end
    cusv
end

function Jisaq.apply!(cusv::Statevector{<:CuArray}, u::U2)
    function k(a, loc, u1,u2,u3,u4)
        i = threadIdx().x - 1
        j = blockIdx().x - 1
        idx = 1024j + i
        lm1 = loc - 1
        idx1 = bit_insert(idx, 1 << lm1 )
        idx2 = idx1 ⊻ 1 << lm1
        idx1 += 1
        idx2 += 1
        x,y = a[idx1],a[idx2]
        a[idx1] = u1*x + u2*y
        a[idx2] = u3*x + u4*y
        return
    end
    arr = cusv.vec
    loc = u.loc
    u1,u3,u2,u4 = u.mat
    if length(arr) ≤ 1024
        @cuda blocks=1 threads=length(arr)÷2 k(arr, loc, u1,u2,u3,u4)
    else
        @cuda blocks=length(arr)÷(2048) threads=1024 k(arr, loc, u1,u2,u3,u4)
    end
    cusv
end

function Jisaq.apply_diagonal1q!(cusv::Statevector{<:CuArray}, loc::Int, d1,d2)
    function k(a, loc, d1, d2)
        i = threadIdx().x - 1
        j = blockIdx().x - 1
        idx = 1024j + i
        lm1 = loc - 1
        idx1 = bit_insert(idx, 1 << lm1 )
        idx2 = idx1 ⊻ 1 << lm1 + 1
        a[idx1+1] *= d1
        a[idx2] *= d2
        return
    end
    arr = cusv.vec
    if length(arr) ≤ 1024
        @cuda blocks=1 threads=length(arr)÷2 k(arr, loc, d1, d2)
    else
        @cuda blocks=length(arr)÷(2048) threads=1024 k(arr, loc, d1, d2)
    end
    cusv
end

#TODO
#cu
#CX
#I+A
#Rzz
#TimeEvolution

end