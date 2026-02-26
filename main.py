"""
Multi-nail instance segmentation and analysis pipeline.

Refer to Context.md for project details.
"""

import json
import logging
from src.analyze import NailAnalyzer
from src.calibration import MeasurementCalibrator
from config import LOG_LEVEL, LOG_FORMAT, MODEL_PATH

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for nail analysis pipeline."""
    try:
        # Example 1: Basic analysis with default calibration
        analyzer = NailAnalyzer(MODEL_PATH)

        # Example 2: Custom calibration (uncomment to use)
        # You can set calibration using:
        # - Default value: MeasurementCalibrator()  # Uses ~200 DPI
        # - Custom DPI: MeasurementCalibrator(pixels_per_mm=6.5)  # ~250 DPI
        # - From reference: calibrator = MeasurementCalibrator()
        #                  calibrator.set_calibration_from_reference(
        #                      reference_pixel_length=150,  # pixels
        #                      reference_real_length=30,    # mm
        #                  )
        #                  analyzer = NailAnalyzer(MODEL_PATH, calibrator)

        # Analyze image
        image_path = "images/test.jpeg"
        results = analyzer.analyze(image_path)

        # Print results
        if results:
            print("\n" + "=" * 70)
            print(f"NAIL ANALYSIS RESULTS ({len(results)} nails detected)")
            print("=" * 70)
            
            # Print calibration info
            cal_info = analyzer.calibrator.get_calibration_info()
            print(f"\nCalibration: {cal_info['dpi']:.1f} DPI ({cal_info['pixels_per_mm']:.4f} px/mm)")
            print("-" * 70)
            
            for nail in results:
                print(f"\n✋ Nail {nail['nail_id']}:")
                
                # Show only stable measurements (ratio is consistent across positions/distances)
                #print(f"  Shape Ratio: {nail['aspect_ratio']:.2f}x (length/breadth)")
                #print(f"  Area: {nail['area_mm2']:.1f} mm²")
                
                # Nail bed measurements (free edge excluded)
                # Keys may be mm or px depending on INCLUDE_REAL_WORLD_MEASUREMENTS
                nb_len = nail.get('nail_bed_length_mm') or nail.get('nail_bed_length_px')
                if nb_len is not None:
                    nb_ratio = nail.get('nail_bed_aspect_ratio', 0)
                    nb_area  = nail.get('nail_bed_area_mm2') or nail.get('nail_bed_area_px', 0)
                    nb_unit  = "mm²" if nail.get('nail_bed_area_mm2') else "px²"
                    fe_pres  = nail.get('nail_bed_free_edge_present')
                    fe_conf  = nail.get('nail_bed_free_edge_confidence', 0)
                    fe_label = ("free edge" if fe_pres is True
                                else "short/trimmed" if fe_pres is False
                                else "polished/unknown")
                    print(f"\n  📋 Nail Bed:")
                    print(f"    Length/Width Ratio: {nb_ratio:.3f}  (>1 = portrait, <1 = wide/thumb)")
                    print(f"    Area: {nb_area:.1f} {nb_unit}")
                    print(f"    Free Edge: {fe_label}  (confidence {fe_conf:.2f})")
                
                # Color information
                if nail.get('nail_color_LAB'):
                    L, a, b = nail['nail_color_LAB']
                    print(f"\n  Color LAB: L={L:.1f}, a={a:.1f}, b={b:.1f}")
                
                if nail.get('nail_color_BGR'):
                    B, G, R = nail['nail_color_BGR']
                    hex_color = nail.get('nail_color_HEX', 'N/A')
                    print(f"  Color BGR: B={B}, G={G}, R={R} ({hex_color})")
                
                # Medical color analysis (if available)
                if nail.get('color_analysis'):
                    analysis = nail['color_analysis']
                    dev = analysis['deviation_from_normal']
                    print(f"  Deviation: L{dev['L_deviation']:+.1f}, a{dev['a_deviation']:+.1f}, b{dev['b_deviation']:+.1f}")
                    
                    # Display nail-skin comparison if available
                    has_skin_ref = analysis.get('has_skin_reference', False)
                    if has_skin_ref and 'relative_metrics' in analysis:
                        rel = analysis['relative_metrics']
                        print(f"\n  📊 Nail vs Skin:")
                        print(f"    Skin L: {rel['skin_reference_L']:.0f}")
                        print(f"    Δ Lightness: {rel['delta_L']:+.0f}")
                        print(f"    Δ Yellowness: {rel['delta_b']:+.0f}")
                    
                    # Display color screening status
                    if 'screening_flags' in analysis:
                        color_flags = [f for f in analysis['screening_flags'] 
                                     if 'color' in f.get('condition', '').lower() or 
                                     'pale' in f.get('condition', '').lower() or
                                     'yellow' in f.get('condition', '').lower() or
                                     'blue' in f.get('condition', '').lower()]
                        
                        if color_flags:
                            color_flag = color_flags[0]
                            condition = color_flag.get('condition', 'Unknown')
                            severity = color_flag.get('severity', 'unknown')
                            
                            if severity == 'none':
                                print(f"\n  🎨 Color: Normal")
                            else:
                                # Remove "Color:" prefix if present and simplify
                                display_condition = condition.replace('Color: ', '').replace('Color screening: ', '')
                                print(f"\n  🎨 Color: {display_condition}")
                    
                    # Display texture status
                    if 'texture_screening' in analysis:
                        tex = analysis['texture_screening']
                        if tex.get('status') == 'completed':
                            result = tex.get('screening_result', 'unknown')
                            if result == 'normal':
                                print(f"\n  🔬 Texture: Normal")
                            else:
                                print(f"\n  🔬 Texture: {result.replace('_', ' ').title()}")
                    
                    # Display screening summary
                    summary = analysis.get('screening_summary', 'normal')
                    if summary == 'recommend_health_check':
                        print(f"\n  ⚠️  Screening: Recommend routine health check")
                    else:
                        print(f"\n  ✓ Screening: Normal")
                
            print("\n" + "=" * 70 + "\n")
        else:
            logger.warning("No nails detected or analysis failed")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
