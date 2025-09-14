
# Salient-Bezier CutMix (SBC): GeoTIFF-aware data augmentation for semantic segmentation

Salient-Bezier CutMix (SBC) is a **geospatially aware** augmentation that blends salient source content into a target raster **while preserving GeoTIFF metadata**. It couples **class-activation saliency** with **Bezier-smoothed masks** to paste *semantically meaningful* regions (not just rectangles), and ships with a rectangular CutMix baseline for apples-to-apples comparisons.

---

## Salient Bezier CutMix 

Classical CutMix uses random rectangles; they’re simple but often paste irrelevant background. SBC instead:

- Finds **salient regions** via Grad-CAM (or any CAM-like map) on a **normalized** copy of the source.
- Converts high-activation blobs into **smooth, closed Bezier polygons**, then rasterizes a **soft mask**.
- **Blends in the original (unnormalized) radiometric space**, so pixel values stay physically plausible.
- Applies the **same geometric augmentation to all branches** (normalized-source, raw-source, target) → no misalignment.
- Writes outputs as **GeoTIFF** (image + label), preserving CRS, transform, and shape.

You can run SBC and rectangle CutMix **side-by-side** to produce two augmented sets for controlled experiments.

---


