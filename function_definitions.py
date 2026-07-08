import numpy as np
from skimage import measure
from skimage.morphology import closing, disk, remove_small_objects

def number_of_objects(labeled_mask: np.ndarray) -> int:
    return len(np.unique(labeled_mask)) - 1



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



def merge_touching_masks(labeled_mask: np.ndarray, footprint_size: int = 1) -> np.ndarray:
    binary_mask = labeled_mask > 0

    footprint = disk(footprint_size)
    merged_masks = closing(binary_mask, footprint)

    new_labeled_mask = measure.label(merged_masks)

    return new_labeled_mask
    



def remove_small_labels(labeled_mask: np.ndarray, min_size: int = 50, relabel: bool = True, connectivity: int = 1) -> np.ndarray:
    cleaned = remove_small_objects(labeled_mask, min_size=min_size)
    if relabel:
        cleaned = measure.label(cleaned, connectivity=connectivity)
    return cleaned