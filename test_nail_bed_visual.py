"""Visual test: extract nail bed + free edge from a static image and display.

Uses the full current pipeline (YOLO segmentation → extract_nail_bed_overlay_data)
and renders an annotated image showing:
  • Blue overlay  – full nail mask
  • Green overlay – nail bed (free edge removed)
  • Yellow line   – detected nail bed / free edge boundary
  • Red arrow     – PCA major axis with distal direction
  • Per-nail HUD  – measurements and diagnostics
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from src.geometry_utils import extract_nail_bed_overlay_data
from config import MODEL_PATH, MASK_THRESHOLD, MIN_MASK_PIXELS

# ── Appearance constants ─────────────────────────────────────────────
FULL_COLOR     = (255, 180, 60)   # full nail — blue-ish
FULL_ALPHA     = 0.20
BED_COLOR      = (60, 220, 60)    # nail bed — green
BED_ALPHA      = 0.30
FREE_EDGE_COLOR = (80, 80, 255)   # free edge area — red-ish
FREE_EDGE_ALPHA = 0.35
BOUNDARY_COLOR = (0, 255, 255)    # boundary line — yellow
AXIS_COLOR     = (0, 200, 255)    # PCA axis — orange
AXIS_HALF      = 70               # half-length of drawn axis in pixels
FONT           = cv2.FONT_HERSHEY_SIMPLEX


def blend_mask(canvas, mask, color, alpha):
    """Alpha-blend a coloured mask onto *canvas* in-place."""
    roi = mask > 0
    if not roi.any():
        return
    canvas[roi] = (
        canvas[roi].astype(np.float32) * (1 - alpha)
        + np.array(color, dtype=np.float32) * alpha
    ).astype(np.uint8)


def put_text(img, text, org, scale=0.55, color=(255, 255, 255), thickness=1):
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), FONT, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def annotate_nail(canvas, nail_mask, data, nail_id, text_y):
    """Draw overlays + HUD for one nail. Returns updated text_y."""
    LINE = 24

    # ── Full nail (blue) ──
    blend_mask(canvas, nail_mask, FULL_COLOR, FULL_ALPHA)

    # ── Nail bed (green) ──
    bed_mask = data["nail_bed_mask"]
    blend_mask(canvas, bed_mask, BED_COLOR, BED_ALPHA)

    # ── Free edge highlight (red-ish) — pixels in full mask but not in bed ──
    fe_mask = cv2.subtract(nail_mask, bed_mask)
    if data.get("free_edge_present") is True:
        blend_mask(canvas, fe_mask, FREE_EDGE_COLOR, FREE_EDGE_ALPHA)

    # ── PCA axis + distal arrow ──
    centroid   = data.get("centroid")
    major_axis = data.get("major_axis")
    minor_axis = data.get("minor_axis")
    if centroid is not None and major_axis is not None:
        cx, cy = int(centroid[0]), int(centroid[1])
        dx, dy = major_axis
        pt1 = (int(cx - dx * AXIS_HALF), int(cy - dy * AXIS_HALF))
        pt2 = (int(cx + dx * AXIS_HALF), int(cy + dy * AXIS_HALF))
        cv2.line(canvas, pt1, pt2, AXIS_COLOR, 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 4, AXIS_COLOR, -1)

        # Arrow toward distal end
        eid = data.get("end_is_distal", True)
        arrow_dir = major_axis if eid else -major_axis
        arrow_tip  = (int(cx + arrow_dir[0] * AXIS_HALF * 0.85),
                      int(cy + arrow_dir[1] * AXIS_HALF * 0.85))
        arrow_base = (int(cx + arrow_dir[0] * AXIS_HALF * 0.50),
                      int(cy + arrow_dir[1] * AXIS_HALF * 0.50))
        cv2.arrowedLine(canvas, arrow_base, arrow_tip, BOUNDARY_COLOR, 2, tipLength=0.4)
        put_text(canvas, "distal", arrow_tip, scale=0.35, color=BOUNDARY_COLOR)

    # ── Boundary line ──
    bp = data.get("boundary_proj")
    if bp is not None and data.get("free_edge_present") is True and centroid is not None:
        half_w = data.get("full_nail_width_px", data["width_px"]) / 2.0
        bxy = centroid + bp * major_axis
        pt1b = tuple((bxy + half_w * minor_axis).astype(int))
        pt2b = tuple((bxy - half_w * minor_axis).astype(int))

        eid = data.get("end_is_distal", True)
        correct_side = (bp > 0) if eid else (bp < 0)
        line_color = BOUNDARY_COLOR if correct_side else (0, 0, 255)
        cv2.line(canvas, pt1b, pt2b, line_color, 2, cv2.LINE_AA)

    # ── HUD text ──
    fe_present = data.get("free_edge_present")
    fe_conf    = data.get("free_edge_confidence", 0.0)
    fe_label   = ("FREE EDGE" if fe_present is True
                  else "trimmed" if fe_present is False
                  else "unknown")
    conf_pct   = f"{fe_conf * 100:.0f}%"

    color = (0, 255, 100) if fe_present else (180, 180, 180)
    put_text(canvas, f"Nail {nail_id}  [{fe_label} {conf_pct}]",
             (14, text_y), color=color)
    text_y += LINE

    # Orientation source
    has_brightness = data.get("brightness_method_used", False)
    n_meth = data.get("n_methods", 0)
    orient_src = "brightness" if has_brightness else (
        "geometry" if n_meth >= 2 else "color-cue"
    )
    eid = data.get("end_is_distal")
    eid_str = "distal=MAX proj" if eid else "distal=MIN proj"
    put_text(canvas, f"  orient: {eid_str}  [{orient_src}]",
             (14, text_y), scale=0.42, color=(200, 200, 100))
    text_y += LINE

    # Method count
    if n_meth > 0:
        mc = ((0, 255, 100) if n_meth >= 3
              else (0, 200, 255) if n_meth >= 2
              else (100, 100, 255))
        put_text(canvas, f"  methods agreed: {n_meth}/5",
                 (14, text_y), scale=0.42, color=mc)
        text_y += LINE

    # Measurements
    length = data.get("length_px", 0)
    width  = data.get("width_px", 0)
    area   = data.get("area_px", 0)
    ratio  = length / width if width > 0 else 0
    put_text(canvas, f"  L={length:.0f}  W={width:.0f}  ratio={ratio:.2f}  area={area:.0f}",
             (14, text_y), scale=0.40, color=(200, 200, 200))
    text_y += LINE + 4

    return text_y


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visual nail-bed / free-edge test")
    parser.add_argument("image", nargs="?", default="images/test.jpeg",
                        help="Path to input image (default: images/test.jpeg)")
    parser.add_argument("--save", default="test_nail_bed_visual_output.png",
                        help="Output filename (default: test_nail_bed_visual_output.png)")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip cv2.imshow (headless)")
    parser.add_argument("--debug", action="store_true",
                        help="Print pipeline debug messages")
    args = parser.parse_args()

    # Load image
    img_path = args.image
    image = cv2.imread(img_path)
    if image is None:
        print(f"ERROR: cannot read image '{img_path}'")
        sys.exit(1)
    print(f"Image: {img_path}  ({image.shape[1]}x{image.shape[0]})")

    # Run YOLO segmentation
    model = YOLO(MODEL_PATH)
    results = model(img_path)[0]
    if results.masks is None:
        print("No nails detected.")
        sys.exit(0)

    masks = results.masks.data.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    print(f"Detected {len(masks)} nail(s)")

    # Precompute LAB frame once — must be float32; the pipeline arithmetic
    # (gradient thresholds, FREE_EDGE_MIN_L=155.0, etc.) depends on it.
    lab_frame = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Prepare canvas (copy of original)
    canvas = image.copy()
    text_y = 30

    for i, (raw_mask, conf) in enumerate(zip(masks, confs), start=1):
        # Resize mask to image dims and binarise
        mask = cv2.resize(raw_mask, (image.shape[1], image.shape[0]))
        mask = (mask > MASK_THRESHOLD).astype(np.uint8) * 255

        if mask.sum() < MIN_MASK_PIXELS:
            print(f"  Nail {i}: skipped (mask too small)")
            continue


        # Run full pipeline and print all relevant outputs
        data = extract_nail_bed_overlay_data(
            image, mask,
            is_polished=False,
            debug=args.debug,
            lab_frame=lab_frame,
        )

        # Draw overlays + HUD
        text_y = annotate_nail(canvas, mask, data, nail_id=i, text_y=text_y)

        fe = data.get("free_edge_present")
        fe_c = data.get("free_edge_confidence", 0)
        n_m = data.get("n_methods", 0)
        br = data.get("brightness_method_used", False)
        length = data.get("length_px", 0)
        width = data.get("width_px", 0)
        area = data.get("area_px", 0)
        ratio = length / width if width > 0 else 0
        boundary_proj = data.get("boundary_proj")
        trimmed = (boundary_proj is None and fe is False)
        print(f"  Nail {i}: conf={conf:.2f}  free_edge={fe}  fe_conf={fe_c:.2f}  methods={n_m}  brightness={br}")
        print(f"    Length: {length:.1f} px, Width: {width:.1f} px, Area: {area:.1f} px^2, Ratio: {ratio:.2f}")
        print(f"    Boundary proj: {boundary_proj}")
        if trimmed:
            print(f"    [SanityCheck] Trimmed fallback used (no boundary detected or geometry extreme)")
        if fe is None:
            print(f"    [SanityCheck] Free edge presence could not be determined.")
        if fe is False and boundary_proj is None:
            print(f"    [SanityCheck] No free edge detected, full mask used.")

    # Save output
    cv2.imwrite(args.save, canvas)
    print(f"\nSaved annotated image → {args.save}")

    # Display
    if not args.no_display:
        # Resize for display if very large
        h, w = canvas.shape[:2]
        max_dim = 1200
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            canvas = cv2.resize(canvas, (int(w * scale), int(h * scale)))
        cv2.imshow("Nail Bed + Free Edge Extraction", canvas)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
