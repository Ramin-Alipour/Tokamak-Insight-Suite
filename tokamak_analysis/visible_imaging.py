"""Visible plasma image intensity and region-of-interest analysis."""
import numpy as np
def analyze_visible_image(image, *, roi=None, dark_frame=None, flat_field=None):
    im=np.asarray(image,dtype=float)
    if im.ndim not in (2,3) or not np.all(np.isfinite(im)): raise ValueError("Image must be finite 2-D or 3-D array")
    corrected=im.copy()
    if dark_frame is not None:
        d=np.asarray(dark_frame,dtype=float)
        if d.shape!=im.shape or not np.all(np.isfinite(d)): raise ValueError("Dark frame must match image")
        corrected-=d
    if flat_field is not None:
        f=np.asarray(flat_field,dtype=float)
        if f.shape!=im.shape or not np.all(np.isfinite(f)) or np.any(f==0): raise ValueError("Flat field must match image and be nonzero")
        corrected=corrected/f
    region=corrected
    if roi is not None:
        y0,y1,x0,x1=map(int,roi)
        if not (0<=y0<y1<=im.shape[0] and 0<=x0<x1<=im.shape[1]): raise ValueError("ROI out of bounds")
        region=corrected[y0:y1,x0:x1,...]
    return {"corrected_image":corrected,"roi_mean_intensity":float(np.mean(region)),"roi_max_intensity":float(np.max(region)),"roi_integrated_intensity":float(np.sum(region)),"roi_shape":region.shape,"caveat":"Absolute emissivity requires radiometric calibration, optical response, and geometric interpretation."}

def pixel_to_object_coordinate(pixel_index, reference_pixel, pixel_pitch_m, object_distance_m, focal_length_m):
    """Pinhole/small-angle calibration: x_object=(pixel-pixel0)*pixel_pitch*distance/focal_length."""
    pix,ref,pitch,distance,focal=np.broadcast_arrays(*[np.asarray(v,dtype=float) for v in (pixel_index,reference_pixel,pixel_pitch_m,object_distance_m,focal_length_m)])
    if any(not np.all(np.isfinite(v)) for v in (pix,ref,pitch,distance,focal)) or np.any(pitch<=0) or np.any(distance<=0) or np.any(focal<=0):
        raise ValueError("Pixel indices finite and optical dimensions positive")
    coord=(pix-ref)*pitch*distance/focal
    return {"object_coordinate_m":coord,"equation":"x_obj=(pixel-pixel_ref)*pixel_pitch*object_distance/focal_length",
            "assumption":"Paraxial pinhole approximation; lens distortion and perspective effects are not corrected."}
