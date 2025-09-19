print("Importing salient_bezier_cutmix")

# Core Libraries
import os
import glob
import random

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# NumPy
import numpy as np

# OpenCV
import cv2

# Visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# PIL
from PIL import Image, ImageDraw, ImageFilter

# TorchVision
from torchvision import transforms, models

# Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# SciPy
from scipy import ndimage
from scipy.special import binom

# RasterIO
import rasterio
from rasterio.transform import Affine

# Albumentations
import albumentations as A

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CustomGeoDataset2(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        #label = self.labels[idx]

        # Read 4-band image with rasterio
        with rasterio.open(img_path) as src:
            image = src.read()  # shape: (bands, height, width)

        # Normalize to 0-1
        image = image.astype('float32')
        for b in range(image.shape[0]):
            band = image[b]
            band = (band - band.min()) / (band.max() - band.min() + 1e-8)
            image[b] = band

        image = torch.tensor(image, dtype=torch.float32)

        # Optional: apply transforms (resize, etc.)
        # Optional resize
        if self.transform:
            image = self.transform(image)
        else:
            image = F.interpolate(image.unsqueeze(0), size=(380,380), mode='bilinear', align_corners=False).squeeze(0)

        return image

def load_random_paths(base_dir, n):
    # Search recursively for .tif files
    pattern = os.path.join(base_dir, "**", "*.tif")
    all_files = glob.glob(pattern, recursive=True)
    return random.sample(all_files, min(n, len(all_files)))

def load_and_preprocess(img_path, size=(256, 256)):
    """Load GeoTIFF with rasterio and normalize"""
    with rasterio.open(img_path) as src:
        img = src.read().astype("float32")  # (C, H, W)

    # Normalize band-wise
    for b in range(img.shape[0]):
        band = img[b]
        img[b] = (band - band.min()) / (band.max() - band.min() + 1e-8)

    # Resize to (C, 256, 256)
    img_tensor = torch.tensor(img).unsqueeze(0)  # (1, C, H, W)
    img_resized = F.interpolate(img_tensor, size=size, mode="bilinear", align_corners=False)
    return img_resized.squeeze(0).numpy()  # (C, H, W)

def contrast_stretch(img, low_pct=2, high_pct=98):
    """
    Apply percentile-based contrast stretch on a 3-channel image.
    img: np.array of shape (H,W,C), float or int
    """
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        low, high = np.percentile(img[:,:,c], (low_pct, high_pct))
        out[:,:,c] = np.clip((img[:,:,c] - low) / (high - low + 1e-8), 0, 1)
    return out

def extract_regions_from_cam(cam_mask, threshold=0.2, min_area=250):
    """
    cam_mask: 2D array normalized 0..1 (grayscale_cam_norm or similar)
    returns: labeled_mask, list of region dicts {label, area, bbox, centroid, coords_mask}
    """
    bw = (cam_mask >= threshold).astype(np.uint8)
    # remove tiny islands
    bw = ndimage.binary_opening(bw, structure=np.ones((3,3))).astype(np.uint8)
    labeled, n = ndimage.label(bw)
    regions = []
    for lab in range(1, n+1):
        coords = np.argwhere(labeled == lab)  # rows (y), cols (x)
        area = coords.shape[0]
        if area < min_area:
            continue
        ys = coords[:,0]
        xs = coords[:,1]
        cy = ys.mean()
        cx = xs.mean()
        y0, x0, y1, x1 = ys.min(), xs.min(), ys.max(), xs.max()
        regions.append({
            "label": lab,
            "area": int(area),
            "bbox": (int(x0), int(y0), int(x1), int(y1)),
            "centroid": (float(cx), float(cy)),
            "coords": coords
        })
    return labeled, regions

def bezier_like_polygon_points(center, area_pixels, img_shape, n_points=16, 
                               min_scale=0.6, max_scale=1.3, jitter_angle=0.5):
    """
    Create polygon points around center. radius scales with sqrt(area).
    center: (cx, cy) in pixel coordinates (x,y)
    area_pixels: area (number of pix) of the high-activation region
    img_shape: (H, W)
    returns: list of (x,y) float points
    """
    H, W = img_shape
    cx, cy = center
    # base radius from area: radius that would have same area if circular
    base_radius = np.sqrt(area_pixels / np.pi)
    # scale to be larger a bit
    mean_radius = max(4.0, base_radius * 1.5)  # at least a few px
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    angles += np.random.uniform(-jitter_angle, jitter_angle, size=n_points)
    radii = np.random.uniform(min_scale, max_scale, size=n_points) * mean_radius
    xs = cx + radii * np.cos(angles)
    ys = cy + radii * np.sin(angles)
    pts = np.stack([xs, ys], axis=1)
    # clip to image bounds
    pts[:,0] = np.clip(pts[:,0], 0, W-1)
    pts[:,1] = np.clip(pts[:,1], 0, H-1)
    return pts.tolist()

def smooth_polygon(points, smoothing_radius=3):
    """
    Given polygon points as list[(x,y)], upsample and apply Gaussian smoothing to coordinates
    to create a smoother curve. Returns list[(x,y)] int suitable for rasterization.
    """
    pts = np.array(points, dtype=np.float32)
    # close loop
    pts_closed = np.vstack([pts, pts[0:1,:]])
    # param t along polygon
    t = np.linspace(0, 1, len(pts_closed))
    # upsample
    t_up = np.linspace(0, 1, max(64, len(pts_closed)*8))
    xs = np.interp(t_up, t, pts_closed[:,0])
    ys = np.interp(t_up, t, pts_closed[:,1])
    # gaussian smooth
    k = max(1, int(smoothing_radius))
    xs = cv2.GaussianBlur(xs.reshape(1,-1).astype(np.float32), (1, k|1), sigmaX=smoothing_radius).flatten()
    ys = cv2.GaussianBlur(ys.reshape(1,-1).astype(np.float32), (1, k|1), sigmaX=smoothing_radius).flatten()
    poly = np.stack([xs, ys], axis=1)
    # convert to ints for rasterization
    poly_int = [(int(round(x)), int(round(y))) for x,y in poly]
    return poly_int

def polygon_to_mask(poly_pts, img_shape, soften_sigma=3):
    """
    Rasterize polygon points to a float mask [0,1], optionally soften with gaussian blur
    poly_pts: list of (x,y) ints
    """
    H, W = img_shape
    mask = np.zeros((H, W), dtype=np.uint8)
    if len(poly_pts) >= 3:
        cv2.fillPoly(mask, [np.array(poly_pts, dtype=np.int32)], color=1)
    mask = mask.astype(np.float32)
    if soften_sigma > 0:
        k = max(3, int(soften_sigma*4)|1)
        mask = cv2.GaussianBlur(mask, (k,k), soften_sigma)
        # normalize to 0..1
        if mask.max() > 0:
            mask = mask / mask.max()
    return mask

bernstein = lambda n, k, t: binom(n, k) * t**k * (1.-t)**(n-k)

def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1
        self.p2 = p2
        self.angle1 = angle1
        self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2 - self.p1)**2))
        self.r = r * d
        self.p = np.zeros((4, 2))
        self.p[0, :] = self.p1[:]
        self.p[3, :] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self, r):
        self.p[1, :] = self.p1 + np.array([self.r * np.cos(self.angle1),
                                           self.r * np.sin(self.angle1)])
        self.p[2, :] = self.p2 + np.array([self.r * np.cos(self.angle2 + np.pi),
                                           self.r * np.sin(self.angle2 + np.pi)])
        self.curve = bezier(self.p, self.numpoints)

def get_curve(points, **kw):
    segments = []
    for i in range(len(points) - 1):
        seg = Segment(points[i, :2], points[i + 1, :2], points[i, 2], points[i + 1, 2], **kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p - np.mean(p, axis=0)
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s), :]

def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points. 
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy) / np.pi + .5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    f = lambda ang: (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang, 1)
    ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x, y = c.T
    return x, y, a

def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7 / n
    a = np.random.rand(n, 2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec >= 200:
        return a * scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec + 1)

def cam_region_to_bezier_mask(cam_mask, threshold=0.1, rad=0.5, edgy=0.8, blur_radius=2):
    """
    Convert high-activation CAM regions into soft Bezier masks.
    cam_mask: 2D np.array normalized [0,1]
    threshold: activation threshold
    blur_radius: Gaussian blur radius for soft boundaries
    Convert high-activation CAM regions into Bezier masks (clipped to image size, with optional blur).
    Returns: list of (polygon_points, polygon_mask)
    """
    H, W = cam_mask.shape
    bw = (cam_mask >= threshold).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []

    for cnt in contours:
        if len(cnt) < 5:
            continue
        pts = cnt.squeeze()
        if pts.ndim != 2 or pts.shape[0] < 5:
            continue

        # Convex hull for smoother boundary
        hull = cv2.convexHull(pts).squeeze().astype(np.float32)

        # --- Get Bezier curve ---
        x, y, _ = get_bezier_curve(hull, rad=rad, edgy=edgy)

        # --- Clip coordinates to valid image bounds ---
        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)

        # --- Round to int for polygon rasterization ---
        poly_points = [(int(xi), int(yi)) for xi, yi in zip(x, y)]

        # --- Create polygon mask ---
        img_mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(img_mask).polygon(poly_points, outline=1, fill=1)
        poly_mask = np.array(img_mask, dtype=np.uint8)

        # --- Apply Gaussian blur (optional feathering) ---
        if blur_radius > 0:
            poly_mask = cv2.GaussianBlur(poly_mask.astype(np.float32), 
                                         (0, 0), blur_radius)
            poly_mask = np.clip(poly_mask, 0, 1)  # normalize back to [0,1]

        results.append((poly_points, poly_mask))

    return results

geom_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.9),
        A.VerticalFlip(p=0.9),
        A.RandomRotate90(p=0.9),
    ],
    additional_targets={
        "raw": "image",
        "target": "image"
    }
)

def apply_cam_and_mask(model, img_array, device="cuda"):
    """Run model -> CAM -> Bezier mask"""
    model.eval()
    target_layers = [model.features[-1][0]]  # last Conv2d
    img_tensor = img_array.unsqueeze(0).to(DEVICE)

    with GradCAM(model=model, target_layers=target_layers) as cam:
        gcam = cam(input_tensor=img_tensor)
        # Normalize CAM
        gcam = (gcam - gcam.min()) / (gcam.max() - gcam.min() + 1e-8)
        gcam = gcam.squeeze()
    
    # Generate mask
    masks = cam_region_to_bezier_mask(gcam, threshold=0.1, rad=0.5, edgy=0.8, blur_radius=2)

    H, W = gcam.shape
    mask_total = np.zeros((H, W), dtype=np.uint8)
    for _, poly_mask in masks:
        mask_total = np.maximum(mask_total, poly_mask)

    # Resize to 256×256
    mask_resized = cv2.resize(mask_total, (256, 256), interpolation=cv2.INTER_LINEAR)

    return mask_resized


def salient_cutmix_pipeline(model, target_drive, source_drive, out_img_drive, out_label_drive, n=500, device="cuda", prefix = 'aug'):
    target_paths = load_random_paths(target_drive, n)
    source_paths = load_random_paths(source_drive, n)

    for i, (target_path, source_path) in enumerate(zip(target_paths, source_paths)):
        # --- Load normalized (for CAM) ---
        s_img = CustomGeoDataset2([source_path], transform=None)[0]  # normalized (C,H,W)
        s_img = s_img.numpy().transpose(1, 2, 0)  # HWC in [0,1]

        # --- Load unnormalized (for blending) ---
        with rasterio.open(source_path) as src:
            s2_img = src.read().transpose(1, 2, 0).astype(np.float32)  # original values
            s2_img = cv2.resize(s2_img, (380, 380), interpolation=cv2.INTER_LINEAR)
            #src_profile = src.profile

        # --- Apply SAME augmentation to both normalized + unnormalized ---
        aug_out = geom_transform(image=s_img, raw=s2_img)
        s_aimg = aug_out["image"]   # normalized → for CAM
        s2_aimg = aug_out["raw"]    # unnormalized → for blending

        # --- Saliency mask from normalized ---
        s_aimg = np.transpose(s_aimg, (2, 0, 1))
        s_tensor = torch.tensor(s_aimg, dtype=torch.float32).to(device)
        mask = apply_cam_and_mask(model, s_tensor, device=device)  # (H,W)
        mask_binary = (mask > 0.25).astype(np.uint8)

        # --- Load and augment target ---
        with rasterio.open(target_path) as tgt:
            profile = tgt.profile
            t_img = tgt.read().transpose(1, 2, 0).astype(np.float32)

        t_aimg = geom_transform(image=t_img)["image"]

        # --- Blend in original value space ---
        mask_expanded = np.expand_dims(mask, axis=-1).astype(np.float32)
        s2_aimg = cv2.resize(s2_img, (256, 256), interpolation=cv2.INTER_LINEAR)
        t_aimg = cv2.resize(t_aimg, (256, 256), interpolation=cv2.INTER_LINEAR)
        mixed_img = mask_expanded * s2_aimg + (1 - mask_expanded) * t_aimg

        # --- Prepare profiles ---
        img_profile = profile.copy()
        img_profile.update({
            "count": mixed_img.shape[2],
            "dtype": "float32"
        })

        label_profile = profile.copy()
        label_profile.update({
            "count": 1,
            "dtype": "float32"
        })

        # --- Save results ---
        out_img_path = os.path.join(out_img_drive, f"{prefix}_image_{i:04d}.tif")
        with rasterio.open(out_img_path, "w", **img_profile) as dst:
            for b in range(mixed_img.shape[2]):
                dst.write(mixed_img[:, :, b], b + 1)

        out_label_path = os.path.join(out_label_drive, f"{prefix}_label_{i:04d}.tif")
        with rasterio.open(out_label_path, "w", **label_profile) as dst:
            dst.write(mask_binary, 1)

        print(f"[{i+1}/{n}] Saved -> {out_img_path}, {out_label_path}")

def random_rectangle_cutmix_mask(h, w, alpha=0.5):
    """
    Classic CutMix rectangle. Returns float32 mask (H,W) with 1 inside the box.
    alpha: Beta(alpha, alpha) controls area. alpha=1.0 ~ uniform variety.
    """
    lam = np.random.beta(alpha, alpha)    # area fraction to keep from target; 1-lam -> source area
    cut_rat = np.sqrt(1.0 - lam)          # side-length ratio for the box
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(0, w)
    cy = np.random.randint(0, h)

    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)

    mask = np.zeros((h, w), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    return mask

def salient_and_rect_dual_save_pipeline(
    model,
    target_drive,
    source_drive,
    out_img_drive,
    out_label_drive,
    n=250,
    device="cuda",
    prefix="aug",
    rect_alpha=1.0,
    inverse = None
):
    """
    For each pair (target, source):
      • Build normalized (for CAM) and unnormalized (for blending) source images.
      • Apply SAME geometric augmentation to (normalized source, unnormalized source, target).
      • Compute saliency mask from normalized source.
      • Generate random rectangular CutMix mask.
      • Produce two blended images: 'salient' and 'rect'.
      • Save image+label GeoTIFFs into separate sub-folders:
          out_img_drive/salient, out_img_drive/rect,
          out_label_drive/salient, out_label_drive/rect
    """
    # sub-folders
    img_sal_dir   = os.path.join(out_img_drive,   "salient")
    img_rect_dir  = os.path.join(out_img_drive,   "rect")
    lab_sal_dir   = os.path.join(out_label_drive, "salient")
    lab_rect_dir  = os.path.join(out_label_drive, "rect")
    for d in [img_sal_dir, img_rect_dir, lab_sal_dir, lab_rect_dir]:
        os.makedirs(d, exist_ok=True)

    target_paths = load_random_paths(target_drive, n)
    source_paths = load_random_paths(source_drive, n)

    for i, (target_path, source_path) in enumerate(zip(target_paths, source_paths)):
        # --- normalized source (for CAM) ---
        s_norm = CustomGeoDataset2([source_path], transform=None)[0]   # (C,H,W), normalized
        s_norm = s_norm.numpy().transpose(1, 2, 0)                     # (H,W,C) in [0,1]

        # --- unnormalized source (for blending) ---
        with rasterio.open(source_path) as src:
            s_raw = src.read().transpose(1, 2, 0).astype(np.float32)   # (Hs,Ws,Cs)

        # match sizes pre-aug if needed
        if s_raw.shape[:2] != s_norm.shape[:2]:
            s_raw = cv2.resize(s_raw, (s_norm.shape[1], s_norm.shape[0]), interpolation=cv2.INTER_LINEAR)

        # --- target (unnormalized) ---
        with rasterio.open(target_path) as tgt:
            profile = tgt.profile
            t_img = tgt.read().transpose(1, 2, 0).astype(np.float32)

        if t_img.shape[:2] != s_norm.shape[:2]:
            t_img = cv2.resize(t_img, (s_norm.shape[1], s_norm.shape[0]), interpolation=cv2.INTER_LINEAR)

        # --- same geometric aug to all three (requires geom_transform with additional_targets) ---
        # geom_transform should be defined elsewhere with additional_targets={"raw":"image","target":"image"}
        out = geom_transform(image=s_norm, raw=s_raw, target=t_img)
        s_aimg_norm = out["image"]    # (H,W,C) normalized for CAM
        s_aimg_raw  = out["raw"]      # (H,W,C) unnormalized for blending
        t_aimg      = out["target"]   # (H,W,C) unnormalized target

        H, W = s_aimg_raw.shape[:2]

        # --- SALIENCY MASK ---
        s_chw = s_aimg_norm.transpose(2, 0, 1)                       # (C,H,W)
        s_tensor = torch.tensor(s_chw, dtype=torch.float32).to(device)
        sal_mask = apply_cam_and_mask(model, s_tensor, device=device).astype(np.float32)
        if sal_mask.shape != (H, W):
            sal_mask = cv2.resize(sal_mask, (W, H), interpolation=cv2.INTER_LINEAR)
        sal_mask = np.clip(sal_mask, 0.0, 1.0)

        # --- RECT MASK ---
        rect_mask = random_rectangle_cutmix_mask(H, W, alpha=rect_alpha).astype(np.float32)

        # --- BLEND (two variants) ---
        sal_mask_exp  = sal_mask[..., None]    # (H,W,1)
        rect_mask_exp = rect_mask[..., None]

        mixed_sal  = (sal_mask_exp  * s_aimg_raw + (1.0 - sal_mask_exp)  * t_aimg).astype(np.float32)
        mixed_rect = (rect_mask_exp * s_aimg_raw + (1.0 - rect_mask_exp) * t_aimg).astype(np.float32)

        mixed_sal = cv2.resize(mixed_sal, (256, 256), interpolation=cv2.INTER_LINEAR)
        mixed_rect = cv2.resize(mixed_rect, (256, 256), interpolation=cv2.INTER_LINEAR)

        # --- INVERSE ---
        if inverse == True:
            sal_mask = cv2.resize((1 - sal_mask[..., None] ), (256, 256), interpolation=cv2.INTER_LINEAR)
            rect_mask = cv2.resize((1 - rect_mask[..., None] ), (256, 256), interpolation=cv2.INTER_LINEAR)
        else :
            sal_mask = cv2.resize(sal_mask, (256, 256), interpolation=cv2.INTER_LINEAR)
            rect_mask = cv2.resize(rect_mask, (256, 256), interpolation=cv2.INTER_LINEAR)

        # --- PROFILES (match output size; adjust transform if resized) ---
        img_profile   = profile.copy()
        label_profile = profile.copy()

        #out_h, out_w = H, W
        out_h, out_w = 256, 256
        in_h, in_w   = profile["height"], profile["width"]

        img_profile.update({
            "height": out_h,
            "width":  out_w,
            "count":  mixed_sal.shape[2],   # both mixed_* have same channel count
            "dtype":  "float32"
        })
        label_profile.update({
            "height": out_h,
            "width":  out_w,
            "count":  1,
            "dtype":  "float32"
        })

        if (out_h, out_w) != (in_h, in_w):
            sx = in_w / float(out_w)
            sy = in_h / float(out_h)
            img_profile["transform"]   = profile["transform"] * Affine.scale(sx, sy)
            label_profile["transform"] = img_profile["transform"]

        # --- SAVE: salient ---
        img_path_sal = os.path.join(img_sal_dir,  f"{prefix}3_image_{i:04d}.tif")
        lab_path_sal = os.path.join(lab_sal_dir,  f"{prefix}3_label_{i:04d}.tif")
        with rasterio.open(img_path_sal, "w", **img_profile) as dst:
            for b in range(mixed_sal.shape[2]):
                dst.write(mixed_sal[:, :, b], b + 1)
        with rasterio.open(lab_path_sal, "w", **label_profile) as dst:
            dst.write((sal_mask > 0.25).astype(np.uint8), 1)

        # --- SAVE: rect ---
        img_path_rect = os.path.join(img_rect_dir, f"{prefix}3_image_{i:04d}.tif")
        lab_path_rect = os.path.join(lab_rect_dir, f"{prefix}3_label_{i:04d}.tif")
        with rasterio.open(img_path_rect, "w", **img_profile) as dst:
            for b in range(mixed_rect.shape[2]):
                dst.write(mixed_rect[:, :, b], b + 1)
        with rasterio.open(lab_path_rect, "w", **label_profile) as dst:
            dst.write((rect_mask > 0.25).astype(np.uint8), 1)

        print(f"[{i+1}/{n}] Saved -> "
              f"{img_path_sal}, {lab_path_sal} | {img_path_rect}, {lab_path_rect}")

if __name__ == "__main__":
    print("Module loaded OK")
