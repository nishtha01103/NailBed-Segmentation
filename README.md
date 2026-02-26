# Nail Segmentation & Analysis Pipeline

Multi-nail instance segmentation and analysis system using YOLOv8-seg for detecting, measuring, and analyzing fingernail characteristics.

## Project Overview

This project detects multiple nails in images, segments them individually, and extracts:
- **Geometric measurements** (length, width, area in pixels)
- **Natural nail color** (LAB and BGR color space values)
- Per-nail structured results for downstream analysis

## Key Features

✅ **Instance Segmentation**: Detects and segments multiple nails independently  
✅ **Robust Color Extraction**: LAB-based color analysis with outlier removal  
✅ **Accurate Measurements**: Rotated bounding box geometry for nail dimensions  
✅ **Medical-Grade Lighting Correction**: Gray-world white balance normalization  
✅ **CPU Optimized**: Efficient processing on CPU-only systems (Intel i5, 8GB RAM)  
✅ **Production Ready**: Type hints, error handling, logging, configuration  
✅ **Modular Design**: Clean separation of concerns across utility modules  

## Project Structure

```
Nail_Segmentation/
├── main.py                   # Entry point with error handling
├── config.py                # Configuration and constants
├── diagnostic.py            # Full segmentation diagnostics
├── geometry_diagnostic.py   # Geometry validation diagnostics
├── reference_measurements.py # Measurement reference calculator
├── requirements.txt         # Python dependencies
├── Context.md              # Project documentation
├── README.md               # This file
├── GEOMETRY_FIX.md         # Geometry validation guide
├── IMPROVEMENTS.md         # Code quality improvements
├── CALIBRATION.md          # Calibration documentation
├── models/
│   └── best.pt             # YOLOv8-seg trained model
├── images/
│   └── test_image.jpg      # Test input image
└── src/
    ├── __init__.py         # Package initialization
    ├── analyze.py          # NailAnalyzer class (main pipeline)
    ├── color_utils.py      # Color extraction with LAB space
    ├── geometry_utils.py   # Geometric measurements with validation
    └── calibration.py      # Real-world unit conversion
```

## Installation & Setup

### 1. Create Virtual Environment
```bash
python -m venv myenv
myenv\Scripts\activate  # Windows
# or: source myenv/bin/activate  # macOS/Linux
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Setup
```bash
python main.py
```

## Configuration

Edit `config.py` to adjust parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MASK_THRESHOLD` | 0.5 | Binary threshold for mask conversion |
| `MIN_MASK_PIXELS` | 1000 | Minimum nail size filter (pixels²) |
| `LAB_L_MIN` | 40 | Shadow removal threshold |
| `LAB_L_MAX` | 230 | Highlight removal threshold |
| `LAB_A_OUTLIER_PERCENTILE` | 5 | Chromaticity outlier rejection (%) |
| `LAB_B_OUTLIER_PERCENTILE` | 5 | Chromaticity outlier rejection (%) |
| `NORMALIZE_WHITE_BALANCE` | True | Enable gray-world white balance normalization (medical) |
| `DEFAULT_PIXELS_PER_MM` | 5.0 | Calibration factor (~200 DPI) |
| `USE_AUTOMATIC_CALIBRATION` | False | Auto-estimate DPI from nail dimensions |
| `INCLUDE_REAL_WORLD_MEASUREMENTS` | True | Add mm/cm values to output |
| `LOG_LEVEL` | INFO | Logging verbosity (DEBUG, INFO, WARNING) |

## Usage

### Basic Usage
```python
from src.analyze import NailAnalyzer

analyzer = NailAnalyzer("models/best.pt")
results = analyzer.analyze("images/test_image.jpg")

for nail in results:
    print(f"Nail {nail['nail_id']}:")
    print(f"  Length: {nail['length_px']} px  |  {nail['length_mm']} mm  |  {nail['length_cm']} cm")
    print(f"  Width: {nail['width_px']} px  |  {nail['width_mm']} mm  |  {nail['width_cm']} cm")
    print(f"  Area: {nail['area_px']} px²  |  {nail['area_mm2']} mm²")
    print(f"  LAB Color: {nail['nail_color_LAB']}")
    print(f"  BGR Color: {nail['nail_color_BGR']}")
```

### With Custom Calibration
```python
from src.analyze import NailAnalyzer
from src.calibration import MeasurementCalibrator

# Option 1: Set DPI directly
calibrator = MeasurementCalibrator(pixels_per_mm=6.5)  # ~250 DPI
analyzer = NailAnalyzer("models/best.pt", calibrator)

# Option 2: Calibrate from reference object
calibrator = MeasurementCalibrator()
calibrator.set_calibration_from_reference(
    reference_pixel_length=150,  # pixels
    reference_real_length=30,    # mm
)
analyzer = NailAnalyzer("models/best.pt", calibrator)

results = analyzer.analyze("images/test_image.jpg")
```

### With Error Handling
```python
try:
    analyzer = NailAnalyzer("models/best.pt")
    results = analyzer.analyze("path/to/image.jpg")
except FileNotFoundError as e:
    print(f"File error: {e}")
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Model error: {e}")
```

## Output Format

Each nail is returned as a dictionary with pixel, relative, and real-world measurements:

```python
{
    "nail_id": 1,
    
    # Pixel measurements
    "length_px": 45.23,
    "width_px": 28.15,
    "area_px": 1089.42,
    
    # Relative measurements (dimensionless)
    "aspect_ratio": 1.61,          # length/width ratio (1.2-3.0 expected)
    "relative_size_percent": 0.08, # nail area as % of total image
    
    # Real-world measurements (mm)
    "length_mm": 9.04,
    "length_cm": 0.90,
    "width_mm": 5.63,
    "width_cm": 0.56,
    "area_mm2": 43.58,
    "area_cm2": 0.44,
    
    # Color information
    "nail_color_LAB": [178.5, -2.1, 15.3],  # [L, a, b]
    "nail_color_BGR": [180, 165, 140]
}
```

**Note:** Real-world measurements (mm, cm, mm², cm²) are automatically calculated using the calibration factor. Adjust `DEFAULT_PIXELS_PER_MM` in `config.py` or provide a custom `MeasurementCalibrator` for accurate results.

## Improvements Implemented

### Code Quality
- ✅ Type hints on all functions
- ✅ Comprehensive docstrings
- ✅ Proper package structure with `__init__.py`
- ✅ Fixed relative import issues

### Error Handling
- ✅ File existence validation
- ✅ Model loading error handling
- ✅ Image reading error handling
- ✅ Graceful failure for edge cases

### Robustness
- ✅ Parameterized thresholds in `config.py`
- ✅ Outlier removal in color extraction (a, b chromaticity)
- ✅ Minimum area validation
- ✅ Pixel count validation

### Logging & Debugging
- ✅ Structured logging with timestamps
- ✅ DEBUG level logs for skipped masks
- ✅ INFO level logs for successful detections
- ✅ WARNING level logs for color extraction failures

### Color Extraction Enhancements
- ✅ Median-based estimation (robust to outliers)
- ✅ L channel filtering (shadow/highlight removal)
- ✅ Percentile-based outlier removal on a, b channels
- ✅ Minimum pixel count validation
- ✅ Graceful failure if insufficient valid pixels
- ✅ Gray-world white balance normalization (medical-grade lighting correction)

### Geometry Measurement
- ✅ Type hints for output
- ✅ Minimum contour area validation
- ✅ Proper float conversion
- ✅ Clear length/width terminology

### Real-World Measurement Calibration ✨ NEW
- ✅ Automatic pixel-to-mm/cm conversion
- ✅ Flexible calibration methods (DPI direct, reference object)
- ✅ Support for automatic DPI estimation from nail size
- ✅ Real-world area calculations (mm²)
- ✅ Detailed calibration information (DPI, px/mm, px/cm)
- ✅ Batch conversion for multiple nails

## Performance Notes

- **Model Loading**: ~1-2 seconds (first run)
- **Image Inference**: ~2-3 seconds (CPU)
- **Color Extraction**: O(n) where n = nail pixels
- **Memory**: ~400MB for typical images

## Geometry Validation (NEW)

Before calibration, pixel measurements are validated to ensure they represent actual nails:

### Validation Steps
1. **Largest contour** - Filters noise and background
2. **Aspect ratio check** - Ensures length/width is 1.2-3.0x (nail-like)
3. **Minimum area** - Filters tiny noise regions

### Check Your Geometry

Run the geometry diagnostic to see which detections pass/fail:
```bash
python geometry_diagnostic.py images/test_image.jpg models/best.pt
```

Shows:
- ✓ Which nails PASS validation
- ❌ Which detections FAIL and why (aspect ratio, area, etc.)
- Estimated DPI for each nail

See [GEOMETRY_FIX.md](GEOMETRY_FIX.md) for detailed troubleshooting.

---

## Measurement Calibration (mm/cm Conversion)

Convert pixel measurements to real-world units using flexible calibration methods.

### How Calibration Works

The calibration factor (`pixels_per_mm`) converts pixel distances to physical units:
- **200 DPI** ≈ 5.0 pixels/mm (default)
- **300 DPI** ≈ 11.8 pixels/mm
- **72 DPI** ≈ 2.83 pixels/mm

### Method 1: Default Calibration (Recommended for General Use)
```python
from src.analyze import NailAnalyzer

# Uses default 200 DPI (~5.0 px/mm)
# Adjust in config.py: DEFAULT_PIXELS_PER_MM
analyzer = NailAnalyzer("models/best.pt")
results = analyzer.analyze("image.jpg")
# Results include: length_mm, length_cm, width_mm, width_cm, area_mm2
```

### Method 2: Custom DPI (When Camera Specs are Known)
```python
from src.analyze import NailAnalyzer
from src.calibration import MeasurementCalibrator

# Set calibration for 250 DPI camera
calibrator = MeasurementCalibrator(pixels_per_mm=9.84)  # 250 DPI
analyzer = NailAnalyzer("models/best.pt", calibrator)
results = analyzer.analyze("image.jpg")
```

### Method 3: Calibrate from Reference Object
Most accurate method - measure a known object in the image:
```python
from src.analyze import NailAnalyzer
from src.calibration import MeasurementCalibrator

calibrator = MeasurementCalibrator()

# If you have a 30mm ruler in the image that spans 150 pixels:
calibrator.set_calibration_from_reference(
    reference_pixel_length=150,  # pixels
    reference_real_length=30,    # mm
)

# Or use centimeters:
calibrator.set_calibration_from_reference(
    reference_pixel_length=240,  # pixels
    reference_real_length=10,    # cm
    unit="cm"
)

analyzer = NailAnalyzer("models/best.pt", calibrator)
results = analyzer.analyze("image.jpg")
```

### Method 4: Automatic Calibration (Experimental)
Auto-estimate DPI from nail size (assumes ~15mm average nail length):
```python
# In config.py, set:
USE_AUTOMATIC_CALIBRATION = True
TYPICAL_NAIL_LENGTH_MM = 15.0  # Adjust if needed

analyzer = NailAnalyzer("models/best.pt")
results = analyzer.analyze("image.jpg")
# DPI is auto-estimated from first detected nail
```

### Get Calibration Information
```python
calibrator = analyzer.calibrator

info = calibrator.get_calibration_info()
print(f"DPI: {info['dpi']:.1f}")
print(f"Pixels/mm: {info['pixels_per_mm']:.4f}")
print(f"Pixels/cm: {info['pixels_per_cm']:.4f}")

# Output: DPI: 200.0 | Pixels/mm: 5.0000 | Pixels/cm: 50.0000
```

### Conversion Functions
```python
from src.calibration import MeasurementCalibrator

calibrator = MeasurementCalibrator(pixels_per_mm=5.0)

# Direct conversions
mm = calibrator.pixel_to_mm(100)      # 20.0 mm from 100 pixels
cm = calibrator.pixel_to_cm(100)      # 2.0 cm from 100 pixels
mm2 = calibrator.pixel_area_to_mm2(500)  # 20.0 mm² from 500 px²
```

### Tips for Accurate Calibration

1. **Use the constant lighting** between calibration and image capture
2. **Capture at consistent distance** from the camera
3. **Use a measurement reference** (ruler, known object) if available
4. **Test calibration accuracy** on multiple images
5. **Adjust `DEFAULT_PIXELS_PER_MM`** in config.py based on your camera

### Common DPI Values
| Device | DPI | px/mm |
|--------|-----|-------|
| Smartphone (modern) | 300-450 | 11.8-17.7 |
| Desktop camera | 72-100 | 2.8-3.9 |
| High-res camera | 200-300 | 7.9-11.8 |
| Default (config) | 200 | 5.0 |

---

## Troubleshooting

### Diagnostic Tools

**Full Segmentation Diagnostics** - Image resolution, mask analysis, threshold effects:
```bash
python diagnostic.py images/test_image.jpg models/best.pt --save-masks
# Creates debug_masks/ with visual overlays
```

**Geometry Validation Diagnostics** - Which detections pass/fail validation:
```bash
python geometry_diagnostic.py images/test_image.jpg models/best.pt
# Shows aspect ratio, area, rejection reasons
```

**Measurement Reference** - Estimate DPI and compare with expected values:
```bash
python reference_measurements.py
# Shows expected pixel measurements at different DPI values
```

### Common Issues

#### "Model not found" error
- Verify `models/best.pt` exists
- Check path in `config.py` MODEL_PATH

#### "Failed to read image" error
- Ensure image path is correct
- Check image is valid JPG/PNG format

#### No nails detected or very few nails
- Run geometry diagnostic to see rejection reasons
- If aspect ratio is the issue, adjust in `config.py`:
  ```python
  MIN_NAIL_ASPECT_RATIO = 1.0   # Lower from 1.2
  MAX_NAIL_ASPECT_RATIO = 4.0   # Raise from 3.0
  ```
- If area is too small, adjust:
  ```python
  MIN_CONTOUR_AREA = 50  # Lower from 100
  ```

#### Measurements look too large (pixel values in 400+)
- Check image resolution: `python diagnostic.py ...`
- Likely image is very large (3000+ pixels)
- Try downsampling image first or adjust `DEFAULT_PIXELS_PER_MM` in `config.py`
- Use `reference_measurements.py` to estimate expected DPI

#### Color extraction returns None
- Image may have unusual lighting
- Adjust `LAB_L_MIN`, `LAB_L_MAX` thresholds in `config.py`
- Check nail is clearly defined in mask (`--save-masks`)

## Dependencies

- `ultralytics` - YOLOv8 framework
- `opencv-python` - Image processing
- `numpy` - Numerical operations
- `scikit-image` - Additional image utilities
- `scikit-learn` - Optional ML utilities

## License & Attribution

Built for nail segmentation and analysis research.
Uses [YOLOv8](https://github.com/ultralytics/ultralytics) by Ultralytics.

## Future Enhancements

- [x] Real-world measurement calibration (mm/cm conversion) ✅ IMPLEMENTED
- [ ] Nail health assessment metrics
- [ ] Batch processing pipeline
- [ ] GPU acceleration support
- [ ] Web API interface
- [ ] Visualization with annotated output

<!-- Removed clinical interpretation details
Removed L-value distribution details
Removed pixel count
Removed clinical assessment warnings 

Next true professional upgrade would be:

Disable webcam auto white balance

Add gray reference calibration

Add nail curvature detection

Add temporal LAB averaging

Add per-nail tracking across frames-->