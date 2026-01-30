import rasterio
from rasterio.features import rasterize
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
from scipy.ndimage import label
import os
from tqdm import tqdm


def clean_coastal_raster(
    raster_path,
    coastline_gdf,
    output_path,
    threshold=0,
    show_progress=True
):
    """
    Remove non-coastal inundation artifacts from a raster by retaining only
    connected flooded regions that are physically connected to the coastline.

    This function is designed for post-processing coastal flood or storm surge
    inundation rasters, where inland rivers, lakes, or depressions may appear
    inundated due to hydraulic connectivity in the raster but are not directly
    connected to the sea.

    Workflow:
    1. Read the input raster and extract inundated cells using a depth threshold.
    2. Rasterize the coastline geometry to identify coastal grid cells.
    3. Label connected inundation regions ("blobs") in the raster.
    4. Retain only those inundation regions that intersect the coastline raster.
    5. Set all other inundated regions to zero and write the cleaned raster
       to disk.

    Parameters
    ----------
    raster_path : str
        Path to the input inundation raster (GeoTIFF).
    coastline_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing coastline geometries.
        Must be in the same CRS as the raster.
    output_path : str
        Path where the cleaned raster will be written.
    threshold : float, optional
        Minimum raster value to consider a cell as inundated.
        Default is 0.

    Outputs
    -------
    A cleaned GeoTIFF raster where only coastal-connected inundation
    is preserved. All other areas are set to zero (nodata = 0).

    Notes
    -----
    - Connectivity is determined using raster connected-component labeling.
    - Inland inundation not connected to the coastline is removed.
    - This method is suitable for storm surge or coastal flooding analyses
      but may not be appropriate for fluvial flood modeling.
    """
    # Read raster
    with rasterio.open(raster_path) as src:
        raster = src.read(1)
        meta = src.meta.copy()

    # Rasterize hollow coastline
    coast_raster = rasterize(
        [(geom, 1) for geom in coastline_gdf.geometry],
        out_shape=raster.shape,
        transform=meta["transform"],
        fill=0,
        all_touched=True,
        dtype=np.uint8
    )

    # Binary inundation mask
    binary = (raster > threshold).astype(np.uint8)

    # Label blobs
    labeled, num_features = label(binary)

    # Keep only blobs touching coastline
    cleaned = np.zeros_like(raster)

    # Iterate through blobs (with progress indicator)
    iterator = range(1, num_features + 1)
    if show_progress:
        iterator = tqdm(
            iterator,
            desc="Cleaning inland inundation",
            unit="blob",
            leave=False
        )

    for blob_id in range(1, num_features + 1):
        blob_mask = (labeled == blob_id)
        if np.any(coast_raster[blob_mask]):
            cleaned[blob_mask] = raster[blob_mask]

    # Save cleaned raster
    meta.update(nodata=0)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(cleaned, 1)


def clip_raster_to_gdf(
    raster_path,
    output_raster,
    gdf,
    nodata=-9999,
    verbose=True
):
    """
    Clip a single GeoTIFF raster using a GeoDataFrame mask.

    Parameters
    ----------
    raster_path : str
        Path to input raster (.tif).
    output_raster : str
        Path to output clipped raster.
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing clipping geometry.
    nodata : int or float, optional
        NoData value to assign to clipped areas.
    verbose : bool, optional
        Print progress messages.
    """

    # --- Open raster ---
    with rasterio.open(raster_path) as src:

        # Ensure CRS match
        if gdf.crs != src.crs:
            if verbose:
                print("Reprojecting shapefile to raster CRS...")
            gdf_proj = gdf.to_crs(src.crs)
        else:
            gdf_proj = gdf

        geometries = gdf_proj.geometry.values

        # --- Clip raster ---
        clipped, transform = mask(
            src,
            geometries,
            crop=True,
            nodata=nodata
        )

        # --- Update metadata ---
        meta = src.meta.copy()
        meta.update({
            "height": clipped.shape[1],
            "width": clipped.shape[2],
            "transform": transform,
            "nodata": nodata
        })

    # --- Write output ---
    with rasterio.open(output_raster, "w", **meta) as dst:
        dst.write(clipped)

    if verbose:
        print(f"✅ Raster clipped successfully: {output_raster}")