using BenchmarkTools

function benchmark(nq::Int, loc_1q, locs_2q)
    res = Dict()
    angle = rand()*2π
    angle2 = rand()*2π
    st = rand_statevec(nq)
    for fun in [X,Y,Z,H]
        res[fun] = @benchmark apply!($st, $fun($loc_1q))
        println(fun, " : ", res[fun])
        flush(stdout)
        GC.gc()
    end
    for fun in [CX]
        res[fun] = @benchmark apply!($st, $fun($locs_2q[1], $locs_2q[2]))
        println(fun, " : ", res[fun])
        flush(stdout)
        GC.gc()
    end
    for fun in [Rx, Ry, Rz]
        res[fun] = @benchmark apply!($st, $fun($loc_1q, $angle))
        println(fun, " : ", res[fun])
        flush(stdout)
        GC.gc()
    end
    for fun in [Rxx, Ryy, Rzz]
        res[fun] = @benchmark apply!($st, $fun($locs_2q[1], $locs_2q[2], $angle))
        println(fun, " : ", res[fun])
        flush(stdout)
        GC.gc()
    end
    for fun in [RyyRxx, RzzRyy]
        res[fun] = @benchmark apply!($st, $fun($locs_2q[1], $locs_2q[2], $angle, $angle2))
        println(fun, " : ", res[fun])
        flush(stdout)
        GC.gc()
    end
    for fun in [copy]
        res[fun] = @benchmark $fun($st)
        println(fun, " : ", res[fun])
        flush(stdout)
        GC.gc()
    end
    res
end