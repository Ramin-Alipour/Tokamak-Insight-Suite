from __future__ import annotations
import numpy as np


def analyze_visible_imaging(frames, time_s, *, pixel_size_m: float, roi=None) -> dict:
    """Extract brightness, centroid and apparent motion from calibrated visible-image sequences."""
    data = np.asarray(frames, dtype=float)
    t = np.asarray(time_s, dtype=float).ravel()
    if data.ndim != 3 or data.shape[0] != t.size or data.shape[0] < 2 or data.shape[1] < 2 or data.shape[2] < 2:
        raise ValueError('frames must have shape (time, y, x) and match at least two time samples.')
    if not np.all(np.isfinite(data)) or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError('Frames must be finite and time_s strictly increasing.')
    if not np.isfinite(pixel_size_m) or pixel_size_m <= 0:
        raise ValueError('pixel_size_m must be positive and finite.')
    if roi is not None:
        if len(roi) != 4:
            raise ValueError('roi must be (x_min, x_max, y_min, y_max).')
        x0, x1, y0, y1 = map(int, roi)
        if x0 < 0 or y0 < 0 or x1 > data.shape[2] or y1 > data.shape[1] or x0 >= x1 or y0 >= y1:
            raise ValueError('ROI lies outside the image or has nonpositive size.')
        data = data[:, y0:y1, x0:x1]
        x_offset, y_offset = x0, y0
    else:
        x_offset = y_offset = 0
    ny, nx = data.shape[1:]
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    cx, cy, brightness, peak = [], [], [], []
    for frame in data:
        positive = np.clip(frame, 0, None)
        total = float(np.sum(positive))
        if total <= np.finfo(float).eps:
            cx.append(np.nan); cy.append(np.nan)
        else:
            cx.append(float(np.sum(xx*positive)/total + x_offset))
            cy.append(float(np.sum(yy*positive)/total + y_offset))
        brightness.append(float(np.mean(frame)))
        peak.append(float(np.max(frame)))
    cx, cy = np.asarray(cx), np.asarray(cy)
    valid = np.isfinite(cx) & np.isfinite(cy)
    speed = np.full(t.size, np.nan)
    if valid.sum() >= 2:
        vx = np.gradient(cx[valid] * pixel_size_m, t[valid])
        vy = np.gradient(cy[valid] * pixel_size_m, t[valid])
        speed[valid] = np.hypot(vx, vy)
    return {'diagnostic': 'Visible imaging', 'time_s': t, 'centroid_x_pixel': cx,
        'centroid_y_pixel': cy, 'centroid_x_m': cx*pixel_size_m, 'centroid_y_m': cy*pixel_size_m,
        'mean_brightness_per_frame': np.asarray(brightness), 'peak_brightness_per_frame': np.asarray(peak),
        'apparent_centroid_speed_m_s': speed,
        'assumptions_and_limits': ['Pixel size must be calibrated in the plane of the image.',
            'Centroid is intensity-weighted and can be biased by background, saturation, occlusion or multiple features.',
            'Apparent image-plane speed is not automatically the plasma flow velocity.']}
