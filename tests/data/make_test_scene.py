#!/usr/bin/env python3
"""Generate a synthetic noisy "path traced" test scene for e2e denoise tests.

Produces color (noisy HDR), albedo and normal PFM images plus a noise-free
color image for visual comparison. Deterministic (fixed seed).
"""

import os
import numpy as np


def write_pfm(path, img):
    """Write a float32 HxWx3 image as a little-endian PFM (bottom-up)."""
    img = np.asarray(img, dtype=np.float32)
    h, w, c = img.shape
    assert c == 3
    with open(path, "wb") as f:
        f.write(b"PF\n")
        f.write(f"{w} {h}\n".encode())
        f.write(b"-1.0\n")  # negative scale = little-endian
        f.write(np.flipud(img).tobytes())


def make_scene(w=512, h=512):
    """Analytic scene: checkered floor, three spheres, sky gradient."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    u = xs / w * 2 - 1
    v = ys / h * 2 - 1

    color = np.zeros((h, w, 3), np.float32)
    albedo = np.zeros((h, w, 3), np.float32)
    normal = np.zeros((h, w, 3), np.float32)

    # Sky gradient
    sky = np.stack([0.4 + 0.4 * (1 - v) * 0.5,
                    0.6 + 0.3 * (1 - v) * 0.5,
                    1.0 + 0.5 * (1 - v) * 0.5], axis=-1)
    color[:] = sky
    albedo[:] = 1.0
    normal[:] = np.array([0, 0, 1], np.float32)

    # Checkered floor in lower half (fake perspective)
    floor = v > 0.1
    fu = u / np.maximum(v, 1e-3) * 2
    fv = 1.0 / np.maximum(v, 1e-3)
    check = ((np.floor(fu) + np.floor(fv)).astype(np.int32) & 1).astype(np.float32)
    floor_alb = np.stack([0.2 + 0.6 * check] * 3, axis=-1)
    floor_alb[..., 2] *= 0.5
    shade = np.clip(1.5 * v, 0, 1)[..., None]
    color[floor] = (floor_alb * shade * 1.2)[floor]
    albedo[floor] = floor_alb[floor]
    normal[floor] = np.array([0, 1, 0], np.float32)

    # Spheres: (cx, cy, r, albedo rgb, emissive scale)
    spheres = [(-0.45, -0.1, 0.30, (0.9, 0.2, 0.15), 1.0),
               (0.35, 0.05, 0.25, (0.2, 0.8, 0.3), 1.0),
               (0.0, -0.45, 0.18, (1.0, 0.9, 0.6), 4.0)]
    light = np.array([0.5, 0.7, 0.5])
    light /= np.linalg.norm(light)
    for cx, cy, r, alb, emis in spheres:
        d2 = (u - cx) ** 2 + (v - cy) ** 2
        mask = d2 < r * r
        nx = (u - cx) / r
        ny = -(v - cy) / r
        nz = np.sqrt(np.clip(1 - nx**2 - ny**2, 0, 1))
        n = np.stack([nx, ny, nz], axis=-1)
        lam = np.clip(n @ light, 0, 1)[..., None]
        alb_arr = np.array(alb, np.float32)
        col = alb_arr * (0.1 + 0.9 * lam) * emis
        color[mask] = col[mask]
        albedo[mask] = alb_arr
        normal[mask] = n[mask]

    return color, albedo, normal


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    rng = np.random.default_rng(42)

    clean, albedo, normal = make_scene()

    # Monte-Carlo-like noise: multiplicative gamma noise (scales with radiance)
    # plus small additive gaussian noise.
    noisy = clean * rng.gamma(shape=4.0, scale=0.25, size=clean.shape).astype(np.float32)
    noisy += rng.normal(0, 0.02, clean.shape).astype(np.float32)
    noisy = np.maximum(noisy, 0).astype(np.float32)

    write_pfm(os.path.join(out_dir, "scene_clean.pfm"), clean)
    write_pfm(os.path.join(out_dir, "scene_color.pfm"), noisy)
    write_pfm(os.path.join(out_dir, "scene_albedo.pfm"), albedo)
    write_pfm(os.path.join(out_dir, "scene_normal.pfm"), normal)
    print("Wrote scene_{clean,color,albedo,normal}.pfm to", out_dir)


if __name__ == "__main__":
    main()
