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

function Jisaq.cu_rand_statevec(nq::Int)
    raw = CUDA.rand(ComplexF64, 2^nq) * 2 .- (1+im) |> statevec
    normalize!(raw)
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
        a[idx1+1] *= d2
        a[idx2] *= d1
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

function Jisaq.apply!(sv::Statevector{<:CuArray}, x::CX)
    function k(v, i, j)
        tidx = threadIdx().x - 1
        bidx = blockIdx().x - 1
        idx = 1024*bidx + tidx
        offset = 1 + 1 << (i-1)
        step = 1 << (j-1)
        _i, _j = minmax(i,j)
        _2_pow_i = 1 << (_i-1)
        _2_pow_j = 1 << (_j-1)
        first = bit_insert(bit_insert(idx, _2_pow_i), _2_pow_j) + offset
        second = first + step
        v[first],v[second] = v[second],v[first]
        return
    end
    nq,v = sv.nq, sv.vec
    i,j = x.ctrl_loc, x.targ_loc
    arr = sv.vec
    if length(arr) ≤ 1024
        @cuda blocks=1 threads=length(arr)÷4 k(arr, i,j)
    else
        @cuda blocks=length(arr)÷(4096) threads=1024 k(arr, i,j)
    end
    sv
end

function Jisaq.apply!(cusv::Statevector{<:CuArray}, ipa::I_plus_A)
    function k(v, step1,_2_pow_maxim1,mask, a1,a2,b,c)
        tidx = threadIdx().x - 1
        bidx = blockIdx().x - 1
        m = 1024*bidx + tidx
        bl = m ÷ (step1 >> 1)
        bm = m % (step1 >> 1)
        l = step1 * bl
        lpk = l + bm
        selector = l & _2_pow_maxim1 == 0
        bc = selector ? b : c
        a12 = selector ? a1 : a2
        idx1 = lpk + 1
        idx2 = lpk ⊻ mask + 1
        x = v[idx1]
        y = v[idx2]
        v[idx1] = x * a12 + y * bc
        v[idx2] = x * bc + y * a12
        return
    end
    a1,a2,b,c = ipa.d1, ipa.d2, ipa.b, ipa.c
    i,j,nq,v = ipa.loc1, ipa.loc2, cusv.nq, cusv.vec
    mask = 2^(i-1) + 2^(j-1)
    mini,maxi = minmax(i,j)
    _2_pow_maxim1 = 2^(maxi-1)
    step1 = 2^mini
    if length(v) ≤ 1024
        @cuda blocks=1 threads=length(v)÷2 k(v, step1,_2_pow_maxim1,mask, a1,a2,b,c)
    else
        @cuda blocks=length(v)÷(2048) threads=1024 k(v, step1,_2_pow_maxim1,mask, a1,a2,b,c)
    end
    cusv
end

function Jisaq.apply!(sv::Statevector{<:CuArray}, x::Rzz)
    function k(v, mask1, mask2, a,b)
        tidx = threadIdx().x - 1
        bidx = blockIdx().x - 1
        idx = 1024*bidx + tidx
        bit1 = idx & mask1 != 0
        bit2 = idx & mask2 != 0
        v[idx+1] *= (bit1 != bit2) ? a : b
        return
    end
    nq,i,j = sv.nq, x.loc1, x.loc2
    v = sv.vec
    theta = x.theta
    a,b = cis(theta/2), cis(-theta/2)
    mask1 = 2^(i-1)
    mask2 = 2^(j-1)
    if length(v) ≤ 1024
        @cuda blocks=1 threads=length(v) k(v, mask1, mask2, a,b)
    else
        @cuda blocks=length(v)÷(1024) threads=1024 k(v, mask1, mask2, a,b)
    end
    sv
end

"""
    CUDA.cu(sv::Statevector)

Copy `sv` to GPU memory. If it is already on GPU, this function just returns `sv`.
"""
CUDA.cu(cusv::Statevector{<:CuArray}) = cusv
CUDA.cu(sv::Statevector) = Statevector(CuArray(sv.vec) )
Jisaq.cpu(cusv::Statevector{<:CuArray}) = Statevector(Array(cusv.vec) )

#TODO
#TimeEvolution

end