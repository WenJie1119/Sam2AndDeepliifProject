import math
import numpy as np
from PIL import Image

# Default postprocessing values
DEFAULT_SEG_THRESH = 120
DEFAULT_NOISE_THRESH = 4

# Values for uint8 label masks
LABEL_UNKNOWN = 50
LABEL_POSITIVE = 200
LABEL_NEGATIVE = 150
LABEL_BACKGROUND = 0
LABEL_CELL = 100
LABEL_BORDER_POS = 220
LABEL_BORDER_NEG = 170
LABEL_BORDER_POS2 = 221
LABEL_BORDER_NEG2 = 171

def imadjust(x, gamma=0.7, c=0, d=1):
    a = x.min()
    b = x.max()
    # Avoid division by zero
    if b == a:
        return x
    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y

def to_array(img, grayscale=False):
    if isinstance(img, Image.Image):
        img = np.asarray(img) if img.mode == 'RGB' else np.asarray(img.convert('RGB'))
    if grayscale and len(img.shape) == 3:
        img = img.max(axis=-1)
    return img

def in_bounds(array, index):
    return index[0] >= 0 and index[0] < array.shape[0] and index[1] >= 0 and index[1] < array.shape[1]

def create_posneg_mask(seg, thresh):
    """Vectorized version of create_posneg_mask without numba."""
    mask = np.full(seg.shape[0:2], LABEL_UNKNOWN, dtype=np.uint8)
    
    # seg is (H, W, 3)
    channel0 = seg[:,:,0].astype(int)
    channel1 = seg[:,:,1].astype(int)
    channel2 = seg[:,:,2].astype(int)
    
    sum_02 = channel0 + channel2
    
    # Condition: seg[y, x, 0] + seg[y, x, 2] > thresh and seg[y, x, 1] <= 80
    cond_main = (sum_02 > thresh) & (channel1 <= 80)
    
    # Positive: seg[y, x, 0] >= seg[y, x, 2]
    cond_pos = channel0 >= channel2
    
    mask[cond_main & cond_pos] = LABEL_POSITIVE
    mask[cond_main & ~cond_pos] = LABEL_NEGATIVE
    
    return mask

def mark_background(mask):
    """
    Mark background pixels.
    This replaces the iterative numba function with a slower but pure python version.
    Or we can try to use some standard image morphology if applicable, 
    but the logic is specific (propagating background from edges).
    This implementation mimics the region growing from edges.
    """
    H, W = mask.shape
    
    # Initialize queue with border UNKNOWN pixels
    queue = []
    
    # Check borders
    # Left & Right
    for i in range(H):
        if mask[i, 0] == LABEL_UNKNOWN:
            mask[i, 0] = LABEL_BACKGROUND
            queue.append((i, 0))
        if mask[i, W-1] == LABEL_UNKNOWN:
            mask[i, W-1] = LABEL_BACKGROUND
            queue.append((i, W-1))
            
    # Top & Bottom
    for j in range(W):
        if mask[0, j] == LABEL_UNKNOWN:
            mask[0, j] = LABEL_BACKGROUND
            queue.append((0, j))
        if mask[H-1, j] == LABEL_UNKNOWN:
            mask[H-1, j] = LABEL_BACKGROUND
            queue.append((H-1, j))
            
    # BFS propagation
    # neighbors 4-connected
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    idx = 0
    while idx < len(queue):
        cy, cx = queue[idx]
        idx += 1
        
        for dy, dx in dirs:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < H and 0 <= nx < W:
                if mask[ny, nx] == LABEL_UNKNOWN:
                    mask[ny, nx] = LABEL_BACKGROUND
                    queue.append((ny, nx))
                    
    # The original second pass (bottom-up) in numba code was likely for iterative separate passes.
    # BFS covers all connected components from edge.
    # But wait, original code also sets UNKNOWN to BACKGROUND if ANY neighbor is BACKGROUND.
    # The while loop in original code:
    # while count > 0:
    #   scan top-down, if UNKNOWN and neighbor BACKGROUND -> set BACKGROUND
    #   scan bottom-up, similar
    # This is effectively flood fill from background. My BFS above is equivalent and O(N).

def compute_cell_mapping(mask, marker, noise_thresh, large_noise_thresh):
    """
    Compute mapping from mask to cells.
    Replaces numba version.
    """
    H, W = mask.shape
    neighbors = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    cells = []
    
    # Iterate over all pixels
    for y in range(H):
        for x in range(W):
            if mask[y, x] != LABEL_BACKGROUND and mask[y, x] != LABEL_CELL:
                # Found a new cell seed
                seeds = [(y, x)]
                count = 1
                count_positive = 1 if mask[y, x] == LABEL_POSITIVE else 0
                count_negative = 1 if mask[y, x] == LABEL_NEGATIVE else 0
                max_marker = marker[y, x] if marker is not None else 0
                
                mask[y, x] = LABEL_CELL
                center_y = y
                center_x = x
                
                # Region growing for this cell
                idx = 0
                while idx < len(seeds):
                    sy, sx = seeds[idx]
                    idx += 1
                    
                    for dy, dx in neighbors:
                        ny, nx = sy + dy, sx + dx
                        if 0 <= ny < H and 0 <= nx < W:
                             if mask[ny, nx] != LABEL_BACKGROUND and mask[ny, nx] != LABEL_CELL:
                                seeds.append((ny, nx))
                                if mask[ny, nx] == LABEL_POSITIVE:
                                    count_positive += 1
                                elif mask[ny, nx] == LABEL_NEGATIVE:
                                    count_negative += 1
                                
                                if marker is not None and marker[ny, nx] > max_marker:
                                    max_marker = marker[ny, nx]
                                
                                mask[ny, nx] = LABEL_CELL
                                center_y += ny
                                center_x += nx
                                count += 1
                
                # Check size thresholds
                if count > noise_thresh and (large_noise_thresh is None or count < large_noise_thresh):
                    avg_cy = int(round(center_y / count))
                    avg_cx = int(round(center_x / count))
                    positive = True if count_positive >= count_negative else False
                    # (count, positive, max_marker, first_x, first_y, center_x, center_y)
                    # Note: first_x/y is just x,y from the outer loop
                    cells.append((count, positive, max_marker, x, y, avg_cx, avg_cy))
                    
    return cells

def create_kde(values, count, bandwidth=1.0):
    gaussian_denom_inv = 1 / math.sqrt(2 * math.pi)
    if len(values) == 0:
        return np.zeros(count, dtype=np.float32), 1.0
        
    max_value = max(values) + 1
    step = max_value / count
    n = len(values)
    h = bandwidth
    h_inv = 1 / h
    kde = np.zeros(count, dtype=np.float32)

    for i in range(count):
        x = i * step
        # Vectorized calculation for speed
        val = (x - values) * h_inv
        total = np.sum(np.exp(-(val*val/2))) * gaussian_denom_inv
        kde[i] = total / (n*h)

    return kde, step

def calculate_default_size_threshold(cell_sizes, resolution='40x'):
    if len(cell_sizes) > 1:
        kde, step = create_kde(np.sqrt(cell_sizes), 500)
        idx = 1
        for i in range(1, len(kde)-1):
            if kde[i] < kde[i-1] and kde[i] < kde[i+1]:
                idx = i
                break
        thresh_sqrt = (idx - 1) * step

        allowed_range_sqrt = (4, 7, 10)
        if resolution == '20x':
            allowed_range_sqrt = (3, 4, 6)
        elif resolution == '10x':
            allowed_range_sqrt = (2, 2, 3)

        if thresh_sqrt < allowed_range_sqrt[0]:
            thresh_sqrt = allowed_range_sqrt[0]
        elif thresh_sqrt > allowed_range_sqrt[2]:
            thresh_sqrt = allowed_range_sqrt[1]

        return round(thresh_sqrt * thresh_sqrt)
    else:
        return 0

def calculate_stain_range(stain):
    nonzero = stain[stain != 0]
    if nonzero.shape[0] > 0:
        return (round(np.percentile(nonzero, 0.1)), round(np.percentile(nonzero, 99.9)))
    else:
        return (0, 0)

def calculate_default_marker_threshold(marker):
    marker_range = calculate_stain_range(marker)
    return round((marker_range[1] - marker_range[0]) * 0.9) + marker_range[0]

def get_cells_info(seg, marker, resolution, noise_thresh, seg_thresh, large_noise_thresh):
    seg = to_array(seg)
    if marker is not None:
        marker = to_array(marker, True)

    mask = create_posneg_mask(seg, seg_thresh)
    mark_background(mask)
    cellsinfo = compute_cell_mapping(mask, marker, noise_thresh, large_noise_thresh)

    defaults = {}
    sizes = np.array([c[0] for c in cellsinfo])
    defaults['size_thresh'] = calculate_default_size_threshold(sizes, resolution)
    if marker is not None:
        defaults['marker_thresh'] = calculate_default_marker_threshold(marker)

    return mask, cellsinfo, defaults

def create_cell_classification(mask, cellsinfo, size_thresh=0, marker_thresh=None, size_thresh_upper=None):
    neighbors = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    border_neighbors = [(0, -1), (-1, 0), (1, 0), (0, 1)]
    
    num_pos = 0
    num_neg = 0
    if marker_thresh is None:
        marker_thresh = 255

    for cell in cellsinfo:
        # cell: (count, positive, max_marker, x, y, center_x, center_y)
        if cell[0] > size_thresh and (size_thresh_upper is None or cell[0] < size_thresh_upper):
            is_pos = True if cell[1] or cell[2] > marker_thresh else False
            if is_pos:
                label = LABEL_POSITIVE
                label_border = LABEL_BORDER_POS
                num_pos += 1
            else:
                label = LABEL_NEGATIVE
                label_border = LABEL_BORDER_NEG
                num_neg += 1

            x = cell[3]
            y = cell[4]
            # Region grow again to paint final labels
            # Note: mask currently has LABEL_CELL where the cells are
            # We overwrite LABEL_CELL with LABEL_POSITIVE/NEGATIVE and add borders
            
            # Since mask was modified in place by compute_cell_mapping to be LABEL_CELL,
            # we need to efficiently find pixels of THIS cell.
            # But wait, compute_cell_mapping set them to LABEL_CELL. 
            # How do we distinguish cells from each other if they are contiguous?
            # Ah, compute_cell_mapping finds SEPARATE connected components because it consumes them.
            # The order in `cellsinfo` corresponds to the order they were found.
            # However, when we re-iterate here, we only have start point (x,y).
            # If we just flood fill LABEL_CELL from (x,y), we get the whole cell.
            # This is safe because adjacent cells would have been merged if they were touching.
            
            mask[y, x] = label_border # Temporarily mark seed
            seeds = [(y, x)]
            
            # Standard python Flood Fill
            idx = 0
            while idx < len(seeds):
                sy, sx = seeds[idx]
                idx += 1
                for dy, dx in neighbors:
                    ny, nx = sy + dy, sx + dx
                    if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                         if mask[ny, nx] == LABEL_CELL:
                             mask[ny, nx] = label # Set to final label
                             seeds.append((ny, nx))
                             # Add border if next to background
                             for bdy, bdx in border_neighbors:
                                 by, bx = ny + bdy, nx + bdx
                                 if 0 <= by < mask.shape[0] and 0 <= bx < mask.shape[1]:
                                     if mask[by, bx] == LABEL_BACKGROUND:
                                         mask[by, bx] = label_border
            
            # Fix the seed pixel itself if it wasn't processed correctly in loop
            mask[y,x] = label 

    return {
        'num_total': num_pos + num_neg,
        'num_pos': num_pos,
        'num_neg': num_neg,
    }

def enlarge_cell_boundaries(mask):
    """Enlarge boundaries by 1 pixel."""
    neighbors = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    H, W = mask.shape
    new_mask = mask.copy()
    
    # This is basically dilation of Border pixels into Background pixels
    # Find all border pixels
    # Using numpy where is faster
    borders_pos = (mask == LABEL_BORDER_POS)
    borders_neg = (mask == LABEL_BORDER_NEG)
    
    # Iterate neighbors
    for dy, dx in neighbors:
        # Shift mask
        # We need to fill BACKGROUND pixels that are neighbors of BORDER
        # This is a bit tricky to vectorize efficiently without shifting 8 times.
        # But 8 shifts is okay.
        
        # Shifted slices
        # src: 0:H, 0:W -> we want to access neighbors.
        # easier: just iterate
        pass
    
    # Python loop is safer for correctness with custom logic
    for y in range(H):
        for x in range(W):
            if mask[y, x] == LABEL_BORDER_POS or mask[y, x] == LABEL_BORDER_NEG:
                val = LABEL_BORDER_POS2 if mask[y, x] == LABEL_BORDER_POS else LABEL_BORDER_NEG2
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] == LABEL_BACKGROUND:
                         # We write to mask directly in original code?
                         # "Enlarge cell boundaries in-place in mask"
                         # Yes. But iterating and writing might affect next pixels.
                         # Original numba code iterates linearly. 
                         # If (y,x) is Border, makes neighbor Border2.
                         # Border2 is NOT Border, so it won't trigger expansion in same pass.
                         mask[ny, nx] = val
                         
    # Second pass to fix labels
    mask[mask == LABEL_BORDER_POS2] = LABEL_BORDER_POS
    mask[mask == LABEL_BORDER_NEG2] = LABEL_BORDER_NEG

def create_final_images(overlay, mask):
    refined = np.zeros_like(overlay)
    H, W = mask.shape
    
    # Vectorized assignment
    # Refined
    refined[mask == LABEL_POSITIVE, 0] = 255
    refined[mask == LABEL_POSITIVE, 1] = 0
    refined[mask == LABEL_POSITIVE, 2] = 0
    
    refined[mask == LABEL_NEGATIVE, 0] = 0
    refined[mask == LABEL_NEGATIVE, 1] = 0
    refined[mask == LABEL_NEGATIVE, 2] = 255
    
    refined[mask == LABEL_BORDER_POS, 0] = 0
    refined[mask == LABEL_BORDER_POS, 1] = 255
    refined[mask == LABEL_BORDER_POS, 2] = 0
    
    refined[mask == LABEL_BORDER_NEG, 0] = 0
    refined[mask == LABEL_BORDER_NEG, 1] = 255
    refined[mask == LABEL_BORDER_NEG, 2] = 0

    # Overlay: paint borders on top
    # LABEL_BORDER_POS -> red (255, 0, 0)
    # LABEL_BORDER_NEG -> blue (0, 0, 255)
    
    overlay[mask == LABEL_BORDER_POS, 0] = 255
    overlay[mask == LABEL_BORDER_POS, 1] = 0
    overlay[mask == LABEL_BORDER_POS, 2] = 0
    
    overlay[mask == LABEL_BORDER_NEG, 0] = 0
    overlay[mask == LABEL_BORDER_NEG, 1] = 0
    overlay[mask == LABEL_BORDER_NEG, 2] = 255
    
    return overlay, refined

def calculate_large_noise_thresh(large_noise_thresh, resolution):
    if large_noise_thresh != 'default':
        return large_noise_thresh
    if resolution == '10x':
        return 1000
    elif resolution == '20x':
        return 4000
    else: # 40x
        return 16000

def compute_final_results(orig, seg, marker, resolution='40x',
                          size_thresh='default',
                          marker_thresh=None,
                          size_thresh_upper=None,
                          seg_thresh=DEFAULT_SEG_THRESH,
                          noise_thresh=DEFAULT_NOISE_THRESH,
                          large_noise_thresh=None):
    
    large_noise_thresh = calculate_large_noise_thresh(large_noise_thresh, resolution)
    mask, cellsinfo, defaults = get_cells_info(seg, marker, resolution, noise_thresh, seg_thresh, large_noise_thresh)

    if size_thresh is None:
        size_thresh = 0
    elif size_thresh == 'default':
        size_thresh = defaults['size_thresh']
    if marker_thresh == 'default':
        marker_thresh = defaults['marker_thresh']

    counts = create_cell_classification(mask, cellsinfo, size_thresh, marker_thresh, size_thresh_upper)
    enlarge_cell_boundaries(mask)
    enlarge_cell_boundaries(mask)
    
    overlay = np.array(orig)
    overlay, refined = create_final_images(overlay, mask)

    scoring = {
        'num_total': counts['num_total'],
        'num_pos': counts['num_pos'],
        'num_neg': counts['num_neg'],
        'percent_pos': round(counts['num_pos'] / counts['num_total'] * 100, 1) if counts['num_total'] > 0 else 0,
        'seg_thresh': seg_thresh,
        'size_thresh': size_thresh,
        'size_thresh_upper': size_thresh_upper,
        'marker_thresh': marker_thresh if marker is not None else None,
    }

    return overlay, refined, scoring
