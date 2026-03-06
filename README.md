# Nail Segmentation & Analysis Pipeline

Real-time and static-image nail segmentation, geometry measurement, and color analysis using YOLOv8-seg. Designed for medical-grade nail morphology screening.

---
<img width="960" height="1280" alt="output4" src="https://github.com/user-attachments/assets/570ceb49-b997-4987-a90c-2213999773a0" />

## Project Overview

The system detects and segments individual fingernails from a webcam feed or a still image, then extracts:

- **Nail-bed geometry** — length, width, area (px and mm), shape ratio, free-edge boundary
- **Full-nail geometry** — stabilized length/width from PCA, directional aspect ratio
- **Color** — CIE LAB and BGR per nail, with SSR illumination normalization
- **Texture** — ridge density, roughness, pitting (configurable sensitivity)
- **Tilt detection** — rejects measurements when the finger is not flat to the camera

Two operating modes share the same underlying analysis modules:

| Mode | Entry point | Input |
|------|------------|-------|
| **Real-time** | `realtime.py` | Webcam |
| **Static image** | `main.py` | JPEG / PNG |

---

## Project Structure

```
Nail_Segmentation/
 main.py                    # Entry point — static image analysis
 realtime.py                # Entry point — live webcam analysis
 config.py                  # All tuneable parameters
 requirements.txt           # Python dependencies
 README.md                  # This file
 models/
    best.pt                # YOLOv8-seg trained model
 images/                    # Test images
 src/
     __init__.py
     analyze.py             # NailAnalyzer class (static pipeline)
     calibration.py         # Pixel  mm/cm conversion
     color_utils.py         # CIE LAB extraction + SSR normalization
     geometry_utils.py      # PCA geometry + nail-bed extraction facade
     texture_analysis.py    # Ridge / roughness / pitting detection
     geometry/              # Nail-bed boundary sub-pipeline
         pipeline.py        # Orchestrates all geometry steps
         boundary_detection.py   # BWC + gradient end selection
         boundary_estimators.py  # Individual boundary methods
         boundary_validation.py  # Multi-method voting & confidence
         bed_mask_builder.py     # Nail-bed mask from accepted boundary
         mask_processing.py      # Morphological helpers
         pca_utils.py            # PCA axis / projection utilities
         axis_orientation.py     # Anatomical axis correction
```

---

## Installation

### 1. Create and activate a virtual environment
```bash
python -m venv myenv
myenv\Scripts\activate          # Windows
# source myenv/bin/activate     # macOS / Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify setup
```bash
python main.py
```

---

## Quick-start

### Static image
```python
from src.analyze import NailAnalyzer

analyzer = NailAnalyzer("models/best.pt")
results = analyzer.analyze("images/test.jpeg")

for nail in results:
    print(f"Nail {nail['nail_id']}")
    print(f"  Length:      {nail['length_mm']:.1f} mm")
    print(f"  Width:       {nail['width_mm']:.1f} mm")
    print(f"  Area:        {nail['area_mm2']:.1f} mm")
    print(f"  Shape ratio: {nail['nail_bed_aspect_ratio']:.2f}x")
    print(f"  LAB:         {nail['nail_color_LAB']}")
```

### Real-time webcam
```python
from realtime import RealtimeNailAnalyzer

analyzer = RealtimeNailAnalyzer("models/best.pt", pixels_per_mm=5.0)
analyzer.run()   # opens webcam window; press Q to quit
```

---

## Output Fields

Every detected nail is returned as a dictionary. The most medically relevant fields are:

| Field | Type | Description |
|-------|------|-------------|
| `nail_id` | int | Detection index in the frame |
| `shape_ratio` | float | **Primary ratio** — nail-bed length/width (nail-bed preferred; full-nail fallback) |
| `aspect_ratio` | float | Full-nail length/width from stabilized PCA (debug reference) |
| `length_mm` | float | Full-nail PCA length, stabilized over 7 frames |
| `width_mm` | float | Full-nail PCA width, stabilized over 7 frames |
| `area_mm2` | float | Full-nail area, stabilized over 5 frames (median) |
| `nail_bed_length_mm` | float | Nail-bed length (free edge excluded) |
| `nail_bed_width_mm` | float | Nail-bed width |
| `nail_bed_area_mm2` | float | Nail-bed area |
| `nail_bed_aspect_ratio` | float | Nail-bed length/width ratio |
| `nail_bed_free_edge_present` | bool | Whether free edge was confidently detected |
| `nail_bed_free_edge_confidence` | float | Confidence of free-edge detection (0–1) |
| `boundary_methods_count` | int | Number of independent methods that agreed on the boundary |
| `boundary_confidence` | float | Combined boundary detection confidence |
| `nail_color_LAB` | list[float] | Median nail color [L, a, b] (OpenCV 0–255 scale) |
| `nail_color_BGR` | list[int] | Median nail color [B, G, R] |
| `nail_color_HEX` | str | Hex color string `#RRGGBB` |
| `tilt_info` | dict | Tilt detection result (`likely_tilted`, `confidence`) |
| `polished` | bool | True if nail appears polished (color analysis skipped) |

---

## Configuration (`config.py`)

### Geometry
| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_NAIL_ASPECT_RATIO` | 0.40 | Minimum length/width (allows wide thumbs) |
| `MAX_NAIL_ASPECT_RATIO` | 3.0 | Maximum length/width |
| `MIN_NAIL_LENGTH_MM` | 3.0 | Reject detections < 3 mm |
| `MAX_NAIL_LENGTH_MM` | 30.0 | Reject detections > 30 mm |
| `MIN_NAIL_AREA_MM2` | 15.0 | Minimum nail plate area |
| `MAX_NAIL_WIDTH_MM` | 22.0 | Rejects wide skin patches |
| `NAIL_BED_NUM_SLICES` | 60 | Projection slices for boundary detection |

### Real-time detection
| Parameter | Default | Description |
|-----------|---------|-------------|
| `REALTIME_DETECTION_CONFIDENCE` | 0.7 | YOLO confidence threshold |
| `REALTIME_MASK_THRESHOLD` | 0.6 | Mask binarization threshold |
| `REALTIME_MIN_MASK_PIXELS` | 2000 | Minimum mask size (px) |
| `CENTROID_DISTANCE_THRESHOLD` | 100 | Max pixel distance for cross-frame nail matching |
| `ASPECT_RATIO_SIMILARITY_THRESHOLD` | 0.3 | Max ratio difference for cross-frame matching |

### Color & lighting
| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENABLE_RETINEX` | True | Single-Scale Retinex illumination normalization on L channel |
| `RETINEX_SIGMA` | 30 | SSR Gaussian blur radius |
| `NORMALIZE_WHITE_BALANCE` | False | Gray-world white balance (disabled — can desaturate) |
| `LAB_L_MIN` | 40 | Shadow removal threshold |
| `LAB_L_MAX` | 230 | Highlight removal threshold |

### Calibration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_PIXELS_PER_MM` | 5.0 |  200 DPI; adjust for your camera |

### Texture analysis
| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENABLE_TEXTURE_ANALYSIS` | True | Ridge / roughness / pitting detection |
| `TEXTURE_SENSITIVITY` | `"conservative"` | `conservative` / `moderate` / `sensitive` |

---

## Calibration

All measurements are converted from pixels using a `pixels_per_mm` factor.

### Method 1: Set DPI directly
```python
from src.calibration import MeasurementCalibrator
calibrator = MeasurementCalibrator(pixels_per_mm=9.84)  # 250 DPI
```

### Method 2: Calibrate from a reference object in the image
```python
calibrator = MeasurementCalibrator()
calibrator.set_calibration_from_reference(
    reference_pixel_length=150,   # pixels in image
    reference_real_length=30,     # mm in reality
)
```

### Common reference values
| Device | DPI | px/mm |
|--------|-----|-------|
| Modern smartphone | 300–450 | 11.8–17.7 |
| Webcam (720p–1080p) | 72–100 | 2.8–3.9 |
| High-res DSLR | 200–300 | 7.9–11.8 |
| Config default | 200 | 5.0 |

---

## Real-time Analyzer — Key Algorithms

### Stable ratio policy
`shape_ratio` always prefers nail-bed geometry when the free edge was confidently detected. When boundary detection fails it falls back to full-nail geometry, preventing measurement jumps between frames.

### Temporal stabilization
All per-nail measurements are smoothed over a rolling window keyed by a **40 px centroid grid** (stable across typical YOLO segmentation jitter):

| Measurement | Window | Aggregation |
|-------------|--------|-------------|
| `length_px` | 7 frames | Mean with 15 % outlier gate |
| `width_px` | 7 frames | Mean with 15 % outlier gate |
| `area_px` | 5 frames | Median with 20 % outlier gate |
| `boundary_proj` | 5 frames | Weighted median |
| `free_edge_present` | 5 frames | Majority vote ( 3/5) |

### Nail-bed boundary detection
The boundary between nail bed and free edge is resolved by four independent estimators whose results are voted on. Boundary presence must remain stable for  3 of the last 5 frames before it is accepted. When no methods agree, or the boundary is absent, full-nail geometry is used as the safe fallback and nail-bed values are clamped to never exceed full-nail dimensions.

**Distal end selection** (which end is the free edge):
1. If bright-width coverage (BWC) differs by  0.12 between ends  use BWC score.
2. If BWC is ambiguous and the gradient energy difference exceeds 0.08  use gradient energy.
3. Otherwise  taper orientation (proximal end is wider).

### Mask completeness check
Before any expensive processing, the coverage of mask pixels along the PCA major axis relative to the rotated bounding box is checked. Masks with coverage < 0.60 are rejected as truncated detections.

### ROI tracker
After pose stabilization, a CSRT tracker follows the nail between YOLO inference frames, reducing latency and CPU load while keeping geometry locked.

---

## Troubleshooting

### No nails detected
- Lower `REALTIME_DETECTION_CONFIDENCE` (try 0.5–0.6).
- Ensure the finger is flat, well-lit, and perpendicular to the camera.
- Check `MIN_NAIL_ASPECT_RATIO` / `MAX_NAIL_ASPECT_RATIO` are not too strict.

### Same nail triggers "NEW NAIL DETECTED" repeatedly
- Centroid jitter is exceeding the 40 px grid — keep the hand more still.
- Check `CENTROID_DISTANCE_THRESHOLD` (default 100 px).

### Measurements jump between frames
- Likely free-edge detection is unstable. The 5-frame majority vote absorbs single-frame failures; if jumps persist, improve lighting and reduce `REALTIME_DETECTION_CONFIDENCE` to get cleaner masks.
- Confirm `ENABLE_RETINEX = True` for consistent illumination.

### Nail-bed ratio > full-nail ratio
- Should not occur — safety clamps are applied after every boundary step.
- Check `boundary_methods_count`: a value of 0 means the anatomical prior triggered the full-nail fallback.

### Color returns None or nail is marked polished
- Nail may be genuinely polished (`polished: True`) — color extraction is intentionally skipped.
- If natural nails are wrongly classified, adjust `NAIL_LAB_*` color validation thresholds in `config.py`.

### Model not found
- Verify `models/best.pt` exists.
- Update `MODEL_PATH` in `config.py` if the model is stored elsewhere.

---

## Dependencies

```
ultralytics       # YOLOv8 inference
opencv-python     # Image processing, CSRT tracker
numpy             # Numerical operations
scipy             # Gaussian filter for boundary smoothing
scikit-image      # Texture / image utilities
scikit-learn      # Optional ML utilities
```

Install with:
```bash
pip install -r requirements.txt
```

---

## License & Attribution

Built for nail morphology research and medical screening.  
Uses [YOLOv8](https://github.com/ultralytics/ultralytics) by Ultralytics.
