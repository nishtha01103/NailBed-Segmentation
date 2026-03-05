"""
Visual test: extract nail bed + free edge from a static image.

Uses geometry-first boundary detection.
Overlays:
  • Blue   – full nail mask
  • Green  – nail bed (free edge removed)
  • Red    – detected free edge region
  • Yellow – boundary line
  • Orange – PCA anatomical axis + distal arrow
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from src.geometry_utils import extract_nail_bed_overlay_data, extract_geometry_nail_bed_with_diagnostics
from config import MODEL_PATH, MASK_THRESHOLD, MIN_MASK_PIXELS


# ────────────────────────────────────────────────────────────────
# Visual constants
# ────────────────────────────────────────────────────────────────
FULL_COLOR = (255, 180, 60)
FULL_ALPHA = 0.20

BED_COLOR = (60, 220, 60)
BED_ALPHA = 0.30

FREE_EDGE_COLOR = (80, 80, 255)
FREE_EDGE_ALPHA = 0.35

BOUNDARY_COLOR = (0, 255, 255)
AXIS_COLOR = (0, 200, 255)

AXIS_HALF = 70
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────
def blend_mask(canvas, mask, color, alpha):
    roi = mask > 0
    if not roi.any():
        return
    canvas[roi] = (
        canvas[roi].astype(np.float32) * (1 - alpha)
        + np.array(color, dtype=np.float32) * alpha
    ).astype(np.uint8)


def put_text(img, text, org, scale=0.55, color=(255, 255, 255), thickness=1):
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), FONT, scale,
                (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale,
                color, thickness, cv2.LINE_AA)


# ────────────────────────────────────────────────────────────────
# Overlay + HUD
# ────────────────────────────────────────────────────────────────
def annotate_nail(canvas, nail_mask, data, nail_id, text_y):

    LINE = 24

    # Full nail
    blend_mask(canvas, nail_mask, FULL_COLOR, FULL_ALPHA)

    # Nail bed
    bed_mask = data["nail_bed_mask"]
    blend_mask(canvas, bed_mask, BED_COLOR, BED_ALPHA)

    # Free edge region
    if data.get("free_edge_present") is True:
        fe_mask = cv2.subtract(nail_mask, bed_mask)
        blend_mask(canvas, fe_mask, FREE_EDGE_COLOR, FREE_EDGE_ALPHA)

    centroid = data.get("centroid")
    major_axis = data.get("major_axis")
    minor_axis = data.get("minor_axis")

    # PCA axis + distal arrow
    if centroid is not None and major_axis is not None:
        cx, cy = int(centroid[0]), int(centroid[1])
        dx, dy = major_axis

        pt1 = (int(cx - dx * AXIS_HALF), int(cy - dy * AXIS_HALF))
        pt2 = (int(cx + dx * AXIS_HALF), int(cy + dy * AXIS_HALF))
        cv2.line(canvas, pt1, pt2, AXIS_COLOR, 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 4, AXIS_COLOR, -1)

        end_is_distal = data.get("end_is_distal", True)
        arrow_dir = major_axis if end_is_distal else -major_axis

        arrow_tip = (int(cx + arrow_dir[0] * AXIS_HALF * 0.85),
                     int(cy + arrow_dir[1] * AXIS_HALF * 0.85))
        arrow_base = (int(cx + arrow_dir[0] * AXIS_HALF * 0.50),
                      int(cy + arrow_dir[1] * AXIS_HALF * 0.50))

        cv2.arrowedLine(canvas, arrow_base, arrow_tip,
                        BOUNDARY_COLOR, 2, tipLength=0.4)
        put_text(canvas, "distal", arrow_tip,
                 scale=0.35, color=BOUNDARY_COLOR)

    # Boundary line
    boundary_proj = data.get("boundary_proj")
    if boundary_proj is not None and centroid is not None:
        full_width = data.get("full_nail_width_px", data["width_px"])
        half_w = full_width / 2.0

        bxy = centroid + boundary_proj * major_axis
        pt1b = tuple((bxy + half_w * minor_axis).astype(int))
        pt2b = tuple((bxy - half_w * minor_axis).astype(int))
        cv2.line(canvas, pt1b, pt2b, BOUNDARY_COLOR, 2, cv2.LINE_AA)

    # HUD
    fe = data.get("free_edge_present")
    fe_conf = data.get("free_edge_confidence", 0.0)

    label = ("FREE EDGE" if fe is True
             else "TRIMMED" if fe is False
             else "UNKNOWN")

    color = (0, 255, 100) if fe else (180, 180, 180)

    put_text(canvas,
             f"Nail {nail_id}  [{label}  {fe_conf*100:.0f}%]",
             (14, text_y), color=color)
    text_y += LINE

    end_is_distal = data.get("end_is_distal")
    put_text(canvas,
             f"  distal at {'MAX proj' if end_is_distal else 'MIN proj'}",
             (14, text_y),
             scale=0.45,
             color=(200, 200, 100))
    text_y += LINE

    length = data.get("length_px", 0)
    width = data.get("width_px", 0)
    area = data.get("area_px", 0)
    ratio = length / width if width > 0 else 0

    put_text(canvas,
             f"  L={length:.0f}  W={width:.0f}  ratio={ratio:.2f}  area={area:.0f}",
             (14, text_y),
             scale=0.42,
             color=(200, 200, 200))
    text_y += LINE + 6

    return text_y


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Geometry-first nail-bed / free-edge test")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--save",
                        default="test_nail_bed_visual_output.png")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"ERROR: cannot read image '{args.image}'")
        sys.exit(1)

    print(f"Image: {args.image}")
    print(f"Resolution: {image.shape[1]}x{image.shape[0]}")

    model = YOLO(MODEL_PATH)
    results = model(args.image)[0]

    if results.masks is None:
        print("No nails detected.")
        sys.exit(0)

    masks = results.masks.data.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    print(f"Detected {len(masks)} nail(s)")

    lab_frame = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    canvas = image.copy()
    text_y = 30

    for i, (raw_mask, conf) in enumerate(zip(masks, confs), start=1):

        mask = cv2.resize(raw_mask,
                          (image.shape[1], image.shape[0]))
        mask = (mask > MASK_THRESHOLD).astype(np.uint8) * 255

        if mask.sum() < MIN_MASK_PIXELS:
            print(f"  Nail {i}: skipped (mask too small)")
            continue


        # Overlay data for visualization
        data = extract_nail_bed_overlay_data(
            image, mask,
            is_polished=False,
            debug=args.debug,
            lab_frame=lab_frame,
        )

        # Diagnostics for logging
        _, _, _, diag = extract_geometry_nail_bed_with_diagnostics(
            image, mask, verbose=args.debug, is_polished=False, lab_frame=lab_frame
        )

        text_y = annotate_nail(canvas, mask, data,
                               nail_id=i, text_y=text_y)

        # Console diagnostics
        print(f"\nNail {i}:")
        print(f"  YOLO conf: {conf:.2f}")
        print(f"  Free edge: {data.get('free_edge_present')}")
        print(f"  Confidence: {data.get('free_edge_confidence'):.2f}")
        print(f"  Boundary proj: {data.get('boundary_proj')}")
        print(f"  Distal at: {'MAX' if data.get('end_is_distal') else 'MIN'}")
        print(f"  Length: {data.get('length_px'):.1f}px")
        print(f"  Width:  {data.get('width_px'):.1f}px")
        print(f"  Ratio:  {(data.get('length_px') / data.get('width_px')) if data.get('width_px') else 0:.2f}")

        # Advanced diagnostics if available
        if diag:
            for key in ["delta_L_region", "L_std", "bright_cov", "region_score", "best_score", "rel_pos", "bed_fraction"]:
                if key in diag:
                    print(f"  {key}: {diag[key]}")

    cv2.imwrite(args.save, canvas)
    print(f"\nSaved annotated image → {args.save}")

    if not args.no_display:
        display = canvas.copy()
        h, w = display.shape[:2]
        if max(h, w) > 1200:
            scale = 1200 / max(h, w)
            display = cv2.resize(display,
                                 (int(w*scale), int(h*scale)))
        cv2.imshow("Nail Bed Extraction", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()