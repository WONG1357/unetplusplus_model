from scipy import ndimage
import numpy as np

def post_process_mask(mask):
    labels, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask
    component_sizes = np.bincount(labels.ravel())
    if len(component_sizes) > 1:
        largest_component_label = component_sizes[1:].argmax() + 1
        processed_mask = (labels == largest_component_label)
        processed_mask = ndimage.binary_fill_holes(processed_mask)
        return processed_mask.astype(np.uint8)
    return mask
