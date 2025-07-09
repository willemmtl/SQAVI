using NetCDF

include("utils.jl")

LEAP_YEARS = getLeapYears();


"""
    preprocessMaxima(maxima)

Store daily precip's maxima per year in a vector of vector.
Convert data units from m to mm.
"""
function preprocessMaxima(maxima_m::Array{Float32, 3})

    # Convert to mm
    maxima = maxima_m .* 1000;

    vectorData = Vector{Vector{Float32}}();
    (M₁, M₂, _) = size(maxima);
    for i = 1:M₁
        for j = 1:M₂
            push!(vectorData, maxima[i, j, :])
        end
    end

    return vectorData

end


"""
concatMacroMaxima(data_dir)

Concatenate macro cells with the right relative position.
Form the full micro grid.
"""
function concatMacroMaxima(data_dir::String)

    microGridSize = getMicroGridSize(data_dir);
    nYears = length(getAllYears(data_dir));
    microGrid = Array{Float32, 3}(undef, (microGridSize..., nYears))

    locations = getAllLocations(data_dir);
    for rlat in locations[:rlats]
        for rlon in locations[:rlons]
            loc = Dict(:rlat => rlat, :rlon => rlon);
            indices = getMicroIndices(loc, data_dir);
            microGrid[indices[:rows], indices[:cols], :] = getMaximaAtLocation(loc, data_dir);
        end
    end

    return microGrid
end


"""
Concatenate coordinates of macro cells to form the micro grid's coordinates.
"""
function concatMacroLoc(data_dir::String)

    microGridSize = getMicroGridSize(data_dir);
    microGridLats = Matrix{Float32}(undef, microGridSize)
    microGridLons = Matrix{Float32}(undef, microGridSize)

    locations = getAllLocations(data_dir);
    for rlat in locations[:rlats]
        for rlon in locations[:rlons]
            loc = Dict(:rlat => rlat, :rlon => rlon);
            indices = getMicroIndices(loc, data_dir);
            coors = getCoordinatesOfLocation(loc, data_dir);
            microGridLats[indices[:rows], indices[:cols]] = coors[:lats];
            microGridLons[indices[:rows], indices[:cols]] = coors[:lons];
        end
    end

    return Dict(
        :lats => microGridLats,
        :lons => microGridLons,
    )
end


"""
    getCoordinatesOfLocation(location, data_dir)

Give lats and lons related to the rlat and rlon ranges in the dataset.
"""
function getCoordinatesOfLocation(location::Dict, data_dir::String)
    
    paths = getAllPaths(data_dir);

    for path in paths
        coor = getCoordinates(path)
        if coor == location
            lats = ncread(path, "lat");
            lons = ncread(path, "lon");
            return Dict(
                :lats => lats,
                :lons => lons,
            )
        end
    end

    println("The given location has not been found in the files.")
end


"""
    getAllYears(data_dir)

List all the years at stake in the dataset.
"""
function getAllYears(data_dir::String)

    periods = String[];
    years = Int64[];

    for path in getAllPaths(data_dir)
        period = getTimePeriod(path);
        if !(period in periods)
            push!(periods, period);
            push!(years, getYears(period)...)
        end
    end

    return years
end


"""
    getMaximaAtLocation(location, folderName)

Get years' maxima of daily precips at given location.
"""
function getMaximaAtLocation(location::Dict{Symbol, String}, folderName::String)

    dailyPrecipPerYear = getDailyPrecipPerYearAtLocation(location, folderName);
    dailyMaximaPerYear = Array{Float32, 3}(undef, (35, 35, length(keys(dailyPrecipPerYear))));

    for (n_y, year) in enumerate(sort(collect(keys(dailyPrecipPerYear))))
        dailyMaximaPerYear[:, :, n_y] = maximum(dailyPrecipPerYear[year], dims=3)[:];
    end

    return dailyMaximaPerYear

end


"""
    getDailyPrecipPerYearAtLocation(location, folderName)

Sort daily precip per year for all years available in the given folderName.

# Arguments
- `location::Dict{Symbol, String}`: rlon's and rlat's ranges.
- `folderName::String`: Name of the folder where the data is stored.
"""
function getDailyPrecipPerYearAtLocation(location::Dict{Symbol, String}, folderName::String)

    dailyPrecipPerYear = Dict{Int, Array{Float32, 3}}()
    files = getPathsWithLocation(location, folderName)
    
    for file in files
        dailyPrecipPerYear = merge(
            dailyPrecipPerYear,
            getFileDailyPrecipPerYear(file)
        )
    end

    return dailyPrecipPerYear
end


"""
    getFileDailyPrecipPerYear(filePath)

Sort the daily precip by year in a dict.
"""
function getFileDailyPrecipPerYear(filePath::String)

    years = getYears(getTimePeriod(filePath));
    dailyPrecipPerYear = Dict{Int, Array{Float32, 3}}()
    hourlyPrecip = ncread(filePath, "CaSR_v3.1_A_PR0_SFC");
    dailyPrecip = computeDailyPrecip(hourlyPrecip);

    firstIndex = 1
    for year in years
        if year in LEAP_YEARS
            lastIndex = firstIndex + 365;
            dailyPrecipPerYear[year] = dailyPrecip[:, :, firstIndex:lastIndex];
            firstIndex = lastIndex + 1;
        else
            lastIndex = firstIndex + 364;
            dailyPrecipPerYear[year] = dailyPrecip[:, :, firstIndex:lastIndex];
            firstIndex = lastIndex + 1;
        end
    end

    @assert firstIndex-1  == size(dailyPrecip, 3);

    return dailyPrecipPerYear

end


"""
    computeDailyPrecip(hourlyPrecip)

Sum hourly precip over 24h to get daily precip.
Raise a warning if some hours are missing.
"""
function computeDailyPrecip(hourlyPrecip::Array{Float32, 3})

    try
        @assert (size(hourlyPrecip, 3) % 24 == 0)
    catch
        println("Le mois n'est pas complet !")
    end
    sizes = size(hourlyPrecip);
    n_days = div(size(hourlyPrecip, 3), 24);

    dailyPrecip = Array{Float32, 3}(undef, (sizes[1], sizes[2], n_days));

    for day = 1:n_days
        first_hour = (day-1)*24 + 1;
        last_hour = (day-1)*24 + 24;
        dailyPrecip[:, :, day] = sum(hourlyPrecip[:, :, first_hour:last_hour], dims=3);
    end

    return dailyPrecip

end;


"""
    getPathsWithLocation(location, folderName)

Get the list of file names containig the given location in the given folder.
"""
function getPathsWithLocation(location::Dict{Symbol, String}, folderName::String)

    pathsWithLocation = String[]
    paths = getAllPaths(folderName);

    for path in paths
        coor = getCoordinates(path)
        if coor == location
            push!(pathsWithLocation, path)
        end
    end

    return pathsWithLocation

end


"""
    getMicroIndices(location, data_dir)

Get indices' range of a given location on the micro grid.
"""
function getMicroIndices(location::Dict, data_dir::String)

    extremeCoors = getMacroGridExtremeCoors(data_dir);
    microGridSize = getMicroGridSize(data_dir);

    macroCellRlats = parse.(Int, String.(split(location[:rlat], "-")));
    macroCellRlons = parse.(Int, String.(split(location[:rlon], "-")));

    ascRowIndex = macroCellRlats .- extremeCoors[:rlatEx][1];
    descRowIndex = microGridSize[1] .- ascRowIndex;
    
    ascColIndex = macroCellRlons .- (extremeCoors[:rlonEx][1] - 1);

    return Dict(
        :rows => descRowIndex[2]:descRowIndex[1],
        :cols => ascColIndex[1]:ascColIndex[2],
    )
end


"""
    getMicroGridSize(data_dir)

Get the size of the 'micro grid' i.e. the grid of the whole dataset.
A 35x35 micro grid corresponds to a macro grid.
"""
function getMicroGridSize(data_dir::String)

    macroGridSize = getMacroGridSize(data_dir);
    return macroGridSize .* 35

end


"""
getMacroGridExtremeCoors(data_dir)

Return extreme coordinates of the macro grid.
i.e. the max and min rlat and rlon.
"""
function getMacroGridExtremeCoors(data_dir::String)

    locations = getAllLocations(data_dir);
    rlatmin = Inf;
    rlatmax = -Inf;
    rlonmin = Inf;
    rlonmax = -Inf;

    for rlat in locations[:rlats]
        rlatstart = parse(Int, rlat[1:3]);
        rlatend = parse(Int, rlat[5:7]);
        if rlatstart < rlatmin
            rlatmin = rlatstart;
        end
        if rlatmax < rlatend
            rlatmax = rlatend;
        end
    end
    for rlon in locations[:rlons]
        rlonstart = parse(Int, rlon[1:3]);
        rlonend = parse(Int, rlon[5:7]);
        if rlonstart < rlonmin
            rlonmin = rlonstart;
        end
        if rlonmax < rlonend
            rlonmax = rlonend;
        end
    end

    return Dict(
        :rlatEx => (rlatmin, rlatmax),
        :rlonEx => (rlonmin, rlonmax),
    )

end


"""
    getMacroGridSize(data_dir)

Get the size of the 'macro grid' i.e. the grid drawn in CaSR's app. 
A macro grid is a 35x35 micro grid.
We consider the macro grid to be a rectangle.
"""
function getMacroGridSize(data_dir::String)

    locations = getAllLocations(data_dir);
    return (
        length(locations[:rlats]),
        length(locations[:rlons]),
    )
    
end


"""
    getAllLocations(data_dir)

Get the rlon and rlat ranges of the whole data folder.
"""
function getAllLocations(data_dir::String)

    paths = getAllPaths(data_dir);

    rlons = String[];
    rlats = String[];
    for path in paths
        coordinates = getCoordinates(path);
        if !(coordinates[:rlon] in rlons)
            push!(rlons, coordinates[:rlon])
        end
        if !(coordinates[:rlat] in rlats)
            push!(rlats, coordinates[:rlat])
        end
    end

    return Dict(
        :rlats => rlats,
        :rlons => rlons,
    )

end


"""
getAllPaths(data_dir)

Get the path of every file in the data directory.

# Arguments
- `data_dir::String`: name of the data directory.
"""
function getAllPaths(data_dir::String)
    
    paths = String[];
    
    for (root, _, files) in walkdir(data_dir)
        if root != "data"
            for file in files
                if file[end-2:end] == ".nc"
                    paths = [paths..., "$root/$file"];
                end
            end
        end
    end
    
    return paths
    
end;