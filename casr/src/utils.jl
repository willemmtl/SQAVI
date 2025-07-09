"""
    getYears(timeRange)

Get the list of years from a period 'yyyy-yyyy'.
Return a variable of type UnitRange{Int64}.
"""
function getYears(timeRange::String)
    (firstY, lastY) = parse.(Int, String.(split(timeRange, "-")));
    return firstY:lastY
end


"""
    getCoordinates(fileName)

Get latitudes and longitudes in rotated pole grid from a file's name.
The file must be in the format '*.rlon111-111*.rlat111-111*.'.
"""
function getCoordinates(fileName::String)
    
    (beforeRLON, afterRLON) = split(fileName, "rlon");
    rlons = afterRLON[1:7];
    (beforeRLAT, afterRLAT) = split(afterRLON, "rlat");
    rlats = afterRLAT[1:7]

    return Dict(
        :rlat => String(rlats),
        :rlon => String(rlons),
    )

end;


"""
    getTimePeriod(fileName)

Get 'year-year' from a file finishing by '*year-year.nc'.
"""
function getTimePeriod(fileName::String)
    return fileName[end-11:end-3]
end


"""
    getLeapYears()

Get the list of leap years between 1900 and 2100.
"""
function getLeapYears()
    leapyears = Int64[];
    years = [i for i = 1900:2100];
    for year in years
        if ((year % 4 == 0) && (!(year % 100 == 0) || (year % 400 == 0)))
            push!(leapyears, year)
        end
    end
    return leapyears
end


"""
    save_array3d(path, A)

Write the 3d array A at the given path.
"""
function save_array3d(path::String, A::Array{Float32, 3})
    open(path, "w") do io
        for d in size(A)
            write(io, Int32(d))
        end
        write(io, A)
    end
end;


"""
    save_matrix(path, A)

Write the matrix A at the given path.
"""
function save_matrix(path::String, A::Matrix{Float32})
    open(path, "w") do io
        for d in size(A)
            write(io, Int32(d))
        end
        write(io, A)
    end
end;


"""
    save_vector(path, vector)

Save the given vector at the given path.
"""
function save_vector(path::String, vector::Vector{Float64})

    open(path, "w") do fichier
        # Écrire chaque élément du vecteur dans le fichier
        for element in vector
            write(fichier, element)
        end
    end

end


"""
    load_array3d(path)

Load the 3d array at the given path.
"""
function load_array3d(path::String)::Array{Float32, 3}
    open(path, "r") do io
        dims = (read(io, Int32), read(io, Int32), read(io, Int32))
        buffer = Vector{Float32}(undef, prod(dims))
        read!(io, buffer)
        reshape(buffer, Tuple(Int.(dims)))  # on reconvertit les dims en Int
    end
end;


"""
    load_matrix(path)

Load the matrix at the given path.
"""
function load_matrix(path::String)::Matrix{Float32}
    open(path, "r") do io
        dims = (read(io, Int32), read(io, Int32))
        buffer = Vector{Float32}(undef, prod(dims))
        read!(io, buffer)
        reshape(buffer, Tuple(Int.(dims)))  # on reconvertit les dims en Int
    end
end;

