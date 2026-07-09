import numpy as np
from skimage import measure, morphology, restoration, exposure
from skimage.morphology import closing, disk, remove_small_objects
from pathlib import Path
from tqdm.auto import tqdm
import tifffile
from scipy.ndimage import median_filter
import pandas as pd


DEFAULT_THRESHOLDS = {"circ1": 0.92, "circ2": 0.956, "ecc": 0.7}


def number_of_objects(labeled_mask: np.ndarray) -> int:
    return len(np.unique(labeled_mask)) - 1



def remove_overexposed_artifacts(image_array: np.ndarray, over_exposure_threshold: int = 60000, disk_size: int = 2) -> np.ndarray:
    image = image_array

    saturated_mask = image > over_exposure_threshold
    footprint = morphology.disk(disk_size)
    dilated_mask = morphology.dilation(saturated_mask, footprint)
    image[dilated_mask] = 0

    return image



def circularity_measures(binary_mask: np.ndarray):
    labeled_mask = measure.label(binary_mask)
    props_list = measure.regionprops(labeled_mask)

    if len(props_list) == 0:
        return np.nan, np.nan, np.nan
    props = max(props_list, key=lambda p: p.area)

    area = props.area
    perimeter = props.perimeter_crofton
    eccentricity = props.eccentricity
    convex_area = props.area_convex

    if perimeter == 0:
        circularity1 = np.nan
    else:
        circularity1 = (4 * np.pi * area)/(perimeter**2)

    if convex_area == 0:
        circularity2 = np.nan
    else:
        circularity2 = area / convex_area
    
    if circularity1 > 1.1:
        print(f"Weirdly high circularity: {circularity1:.3f}")

    return np.clip(circularity1, 0, 1), np.clip(circularity2, 0, 1), np.clip(eccentricity, 0, 1)


# Usead Earlier
def merge_touching_masks(labeled_mask: np.ndarray, footprint_size: int = 1) -> np.ndarray:
    binary_mask = labeled_mask > 0

    footprint = disk(footprint_size)
    merged_masks = closing(binary_mask, footprint)

    new_labeled_mask = measure.label(merged_masks)

    return new_labeled_mask
    



def remove_small_labels(labeled_mask: np.ndarray, max_size: int = 50, relabel: bool = True, connectivity: int = 1) -> np.ndarray:
    cleaned = remove_small_objects(labeled_mask, max_size=max_size)
    if relabel:
        cleaned = measure.label(cleaned, connectivity=connectivity)
    return cleaned




def process_population(population: str, data_root: Path = Path(r".\data")) -> pd.DataFrame:
    """
    A wrapper function for processing cell populations. Example in single_population_analysis.ipynb".

    Input a name of dataset/population and a path to folder.

    Data directory structure:

        data_root/
        ├── cell_images/
        │   ├── merged_BC
        │   ├── {Population Name}
        │   └── ...
        ├── cellpose_masks/
        │   ├── merged_BC
        │   ├── {Population Name}
        │   └── ...
        └── nuclear_masks/
            ├── merged_BC
            ├── {Population Name}
            └── ...
    """
    img_paths = sorted((data_root/"cell_images"/population).rglob("*.ome.tif"))
    cellpose_paths = sorted((data_root/"cellpose_masks"/population).rglob("*.npy"))
    nucleus_paths = sorted((data_root/"nuclear_masks"/population).rglob("*.tif"))

    assert len(img_paths) == len(cellpose_paths) == len(nucleus_paths),\
        f"The numbers of files in given directories do not match. Images: {len(img_paths)}, Cellpose: {len(cellpose_paths)}, Nuclear: {len(nucleus_paths)}."
    

    number_of_images = len(img_paths)
    print(f"{number_of_images} images detected.")


    rows = []

    for n in tqdm(range(number_of_images), f"Processing {population}"):
        tiff = tifffile.imread(img_paths[n])
        cellpose_mask = np.load(cellpose_paths[n], allow_pickle=True).item()['masks']
        nuclear_mask = tifffile.imread(nucleus_paths[n])

        if nuclear_mask.ndim == 3:
            nuclear_mask = nuclear_mask[0]

        # We want to treat splitting/closely agregated cells as singular events, not separate cells
        events_binary_mask = (nuclear_mask * cellpose_mask) > 0
        # events_mask = measure.label(events_binary_mask, connectivity=1)
        events_mask = remove_small_labels(events_binary_mask, max_size=110, connectivity=1, relabel=True)

        number_of_events = number_of_objects(events_mask)


        # Assumes brightfield on the 1st slide and DAPI stain on the 5th one
        brightfield_channel = median_filter(tiff[0], size=3)
        dapi_channel = remove_overexposed_artifacts(median_filter(tiff[4], size=3))

        if number_of_events == 0:
            continue
        
        for EventID in range(number_of_events):
            event_mask = np.where(events_mask==EventID+1, 1, 0).astype(bool)

            # Tracking original Cellpose's IDs
            overlapping_ids = np.unique(cellpose_mask[event_mask])
            overlapping_ids = overlapping_ids[overlapping_ids != 0]
            n_contributing_cells = len(overlapping_ids)

            # Area of the cellpose (cytoplasm) masks over the given nuclear signature
            cellpose_area = np.isin(cellpose_mask, overlapping_ids).sum()

            # Area of the nuclear signature inside of the cellpose mask
            nuc_area = event_mask.sum()

            nuc_by_cell_area = nuc_area / cellpose_area if cellpose_area > 0 else np.nan

            # Mean intensity of the DAPI signal
            nuc_mean = dapi_channel[event_mask].mean() if event_mask.any() else np.nan

            # Circularity measures
            circularity1, circularity2, eccentricity = circularity_measures(event_mask)


            # Pandas DataFrame formatting
            rows.append({
                "Population": population,
                "Image ID": n,
                "Image Path": img_paths[n],
                "Event ID": EventID,
                "N Contributing Cellpose Cells": n_contributing_cells,
                "Cellpose Area": cellpose_area,
                "Nucleus Area": nuc_area,
                "Nucleus/Cellpose": nuc_by_cell_area,
                "Average Nucleus Signal": nuc_mean,
                "Circularity 1": circularity1,
                "Circularity 2": circularity2,
                "Eccentricity": eccentricity
                })
            

    df = pd.DataFrame(rows)     
    df.to_csv(f"{population}.csv", index=False)

    return df



def flag_noncircular(df: pd.DataFrame, thresholds: dict =DEFAULT_THRESHOLDS) -> pd.DataFrame:

    return df[(df["Circularity 1"] < thresholds["circ1"]) |
              (df["Circularity 2"] < thresholds["circ2"]) |
              (df["Eccentricity"] > thresholds["ecc"])]