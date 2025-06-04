# File containing shared parameters for map processing
raster_extensions = (
    ".tif",
    ".tiff",
    ".img",
    ".pix",
    ".rst",
    ".grd",
    ".vrt",
    ".hdf",
    ".h5",
    ".jpeg2000",
)

vector_extensions = (
    ".shp",
    ".sqlite",
    ".gdb",
    ".geojson",
    ".json",
    ".gml",
    ".kml",
    ".tab",
    ".mif",
)

# Add target CRS parameters
target_crs_epsg = 4326
target_crs_str = f"EPSG:{target_crs_epsg}"
