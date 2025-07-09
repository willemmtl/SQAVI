using Distances, NetCDF

include("preprocessing.jl");


"""
Find the cell number that corresponds to Montreal in a given area.
"""
function findMontrealInArea(location::Dict{Symbol, String}, folderName::String)
    
    filePath = getPathsWithLocation(location, folderName)[1];
    lats = ncread(filePath, "lat");
    lons = ncread(filePath, "lon") .- 360;

    return findMontreal(lats, lons);

end;


"""
Find the cell number that corresponds to Montreal.
Montreal coordinates are (45.50884, -73.58781).
"""
function findMontreal(lats::Array{Float32}, lons::Array{Float32})

    cellNum = 0;
    (latmin, lonmin) = (Inf, Inf)
    (M₁, M₂) = size(lats);
    target = [45.50884, -73.58781];
    minDist = Inf;
    
    cell = 0;
    for i = 1:M₁
        for j = 1:M₂
            cell += 1;
            dist = euclidean([lats[i, j], lons[i, j]], target);
            if dist < minDist
                cellNum = cell;
                minDist = dist;
                (latmin, lonmin) = (lats[i, j], lons[i, j]);
            end
        end
    end

    println("Coordonnées du point de plus proche de Montréal : [$latmin, $lonmin]")

    return cellNum

end;