import os
import glob
import geopandas as gpd
import rasterio
import tempfile

from core.raster_ops import clip_raster_to_gdf
from core.raster_ops import clean_coastal_raster

# PATHS
input_folder = r"C:\Users\SHIELD-HTMIRDS\Downloads\Model_builder\Original_rasters_Quezon\rasters" # Input folder containing inundation rasters
output_folder = r"D:\Cleaned_rasters_Quezon" # Output folder for cleaned rasters

province_shp = r"C:\Users\SHIELD-HTMIRDS\Downloads\Model_builder\Shapefiles\Prov_Quezon.shp" # Province boundary shapefile

os.makedirs(output_folder, exist_ok=True)

# LOAD PROVINCE GEOMETRY ONCE
province_gdf = gpd.read_file(province_shp)

# PROCESS RASTERS
raster_files = glob.glob(os.path.join(input_folder, "*.tif"))
print(f"Found {len(raster_files)} raster files")


for raster_path in raster_files:
    fname = os.path.basename(raster_path)
    final_output = os.path.join(output_folder, fname)

    print(f"\nProcessing {fname}...")

    with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
        tmp_clipped = tmp.name

    # 1. Clip raster to province boundary
    clip_raster_to_gdf(
        raster_path=raster_path,
        output_raster=tmp_clipped,
        gdf=province_gdf,
        nodata=-9999,
        verbose=True
    )

    # 2. Prepare hollow coastline from same geometry
    with rasterio.open(tmp_clipped) as src:
        province_proj = province_gdf.to_crs(src.crs)

    coastline_gdf = province_proj.copy()
    coastline_gdf["geometry"] = coastline_gdf.boundary

    # 3. Remove inland (non-coastal) inundation
    clean_coastal_raster(
        raster_path=tmp_clipped,
        coastline_gdf=coastline_gdf,
        output_path=final_output,
        threshold=0,
        show_progress=True
    )

    # Cleanup temporary file
    os.remove(tmp_clipped)

print("\n✅ All rasters processed successfully.")