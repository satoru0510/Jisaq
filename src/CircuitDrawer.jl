export DrawingStyle
abstract type DrawingStyle end

export UnicodeDrawingStyle
"""
    mutable struct UnicodeDrawingStyle <: DrawingStyle

- `prefix_cir=1`: padding before each `circuit` in `Int`
- `postfix_cir=1`: padding after each `circuit` in `Int`
- `prefix_global=1`: padding in start of the drawing in `Int`
- `postfix_global=1`: padding in the end of the drawing in `Int`
"""
mutable struct UnicodeDrawingStyle <: DrawingStyle
    prefix_cir::Int
    postfix_cir::Int
    prefix_global::Int
    postfix_global::Int
end

function UnicodeDrawingStyle(;prefix_cir=1, postfix_cir=1, prefix_global=1, postfix_global=1)
    UnicodeDrawingStyle(prefix_cir, postfix_cir, prefix_global, postfix_global)
end

chmat(x::Type{<:Union{X,Y,Z,H,P0,P1,Rx,Ry,Rz,S,T}}) = hcat(string(nameof(x))...)
chmat(::Type{CX}, n::Int) = (n>0 ? ['┬'; repeat('┼', n-1)...; 'X';] : ['X'; repeat('┼', -n+1)...; '┴';]) |> (x -> reshape(x, 3,1))
function chmat(x::Type{<:Union{Rxx,Ryy,Rzz}}, n::Int)
    c1, c2 = string(nameof(x))[2:3]
    if n>0
        hcat(['R', repeat('┼', n-1)..., '┴'], [c1, repeat('┼', n-1)..., c2])
    else
        hcat(['┬', repeat('┼', -n-1)..., 'R'], [c2, repeat('┼', -n-1)..., c1])
    end
end

function chmat(x::Type{<:Union{RyyRxx, RzzRyy}}, n::Int)
    c1, c2, c3, c4 = string(nameof(x))[[2,3,5,6]]
    if n>0
        hcat(['R', repeat('┼', n-1)..., '┴'], [c3, repeat('─', n-1)..., c4], ['R', repeat('─', n-1)..., ' '], [c1, repeat('┼', n-1)..., c2])
    else
        hcat(['┬', repeat('┼', -n-1)..., 'R'], [c3, repeat('─', -n-1)..., c4], [' ', repeat('─', -n-1)..., 'R'], [c1, repeat('┼', -n-1)..., c2])
    end
end

function chmat_to_drawing(io::IO, chmat::Matrix{Char})
    n,m = size(chmat)
    for i in 1:n
        for j in 1:m
            print(io, chmat[i,j])
        end
        println(io)
    end
end

function append!(canvas::Matrix{Char}, cursor::Vector{Int}, x::Union{X,Y,Z,H,P0,P1,Rx,Ry,Rz,S,T}, style::UnicodeDrawingStyle)
    cm = x |> typeof |> chmat
    loc = locs(x)[1]
    len = size(cm)[2]
    c = cursor[loc]
    nq = length(cursor)
    if c + len - 1 > size(canvas)[2]
        canvas = hcat(canvas, fill('─', nq, c + len - 1 - size(canvas)[2]))
    end
    canvas[loc, c:c+len-1] = cm
    cursor[loc] += len+1
    canvas
end

function append!(canvas::Matrix{Char}, cursor::Vector{Int}, 
        x::Union{Rxx,Ryy,Rzz,RyyRxx,RzzRyy,RxxRzz,RxxRyy,RyyRzz,RzzRxx}, style::UnicodeDrawingStyle)
    loc1, loc2 = locs(x)
    cm = chmat(typeof(x), loc2-loc1)
    len = size(cm)[2]
    _loc1, _loc2 = minmax(loc1, loc2)
    between_locs = _loc1+1 : _loc2-1
    c1,c2 = cursor[[loc1,loc2]]
    nq = length(cursor)
    maxc = maximum(cursor[_loc1 : _loc2])
    if maxc + len - 1 > size(canvas)[2]
        canvas = hcat(canvas, fill('─', nq, maxc + len - 1 - size(canvas)[2]))
    end
    canvas[_loc1:_loc2, maxc:maxc+len-1] = cm
    cursor[_loc1 : _loc2] = repeat([maxc+len+1], length(between_locs)+2)
    canvas
end

function append!(canvas::Matrix{Char}, cursor::Vector{Int}, x::Circuit, style::UnicodeDrawingStyle)
    cmax = maximum(cursor)
    cursor .= cmax
    cursor .+= style.prefix_cir-1
    for gt in x.gates
        canvas = append!(canvas, cursor, gt, style)
    end
    nq = loc_max(x)
    canvas = hcat(canvas, fill('─', nq, style.postfix_cir-1))
    cursor .+= style.postfix_cir
    canvas
end

function draw_unicode(io::IO, cir::Circuit, style::UnicodeDrawingStyle)
    nq = loc_max(cir)
    canvas = init_drawing(nq)
    cursor = fill(style.prefix_global+1, nq)
    canvas = append!(canvas, cursor, cir, style)
    canvas = hcat(canvas, fill('─', nq, style.postfix_global))
    chmat_to_drawing(io, canvas)
end

export draw
"""
    draw([io::IO, ]cir::Circuit, style::UnicodeDrawingStyle)

draw circuit diagram of `cir` with unicode characters
"""
draw(cir::Circuit, style::UnicodeDrawingStyle) = draw_unicode(stdout, cir, style)
draw(io::IO, cir::Circuit, style::UnicodeDrawingStyle) = draw_unicode(io, cir, style)

"""
    draw(cir::Circuit, style::Symbol)

Draw circuit diagram of `cir` with specipied `style`. For now, only `style=:unicode` is supported.
"""
function draw(cir::Circuit, style::Symbol)
    if style == :unicode
        ust = UnicodeDrawingStyle()
        draw_unicode(stdout, cir, ust)
    else
        error()
    end
end

init_drawing(nq::Int) = fill('─', nq,1)