#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
import numpy as np


# Stock euler to quat function
def euler_deg_to_quat(omega_deg, phi_deg, kappa_deg):
    o = math.radians(omega_deg)
    p = math.radians(phi_deg)
    k = math.radians(kappa_deg)

    cy = math.cos(k * 0.5)
    sy = math.sin(k * 0.5)
    cp = math.cos(p * 0.5)
    sp = math.sin(p * 0.5)
    cr = math.cos(o * 0.5)
    sr = math.sin(o * 0.5)

    w = cr * cp * cy + sr * cp * sy
    x = sr * cp * cy - cr * cp * sy
    y = cr * sp * cy + sr * sp * sy
    z = cr * sp * sy - sr * sp * cy
    return w, x, y, z


# I faced some malformed entries, this cleans them and makes sure collums do not ge squashed
def sanitize_numeric_field(value: str):
    floats = re.findall(r"-?\d+\.\d+", value)

    if len(floats) == 0:
        return [value]

    return floats


# Read the orientation fole and put it into a nice mapping
def parse_orientations(path):
    mapping = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            label = parts[0]

            X0 = float(parts[1])
            Y0 = float(parts[2])
            Z0 = float(parts[3])
            omega = float(parts[4])
            phi = float(parts[5])
            kappa = float(parts[6])
            c = float(parts[7])
            cx = float(parts[8])
            cy = float(parts[9])

            qw, qx, qy, qz = euler_deg_to_quat(omega, phi, kappa)

            mapping[label] = {
                "X0": X0,
                "Y0": Y0,
                "Z0": Z0,
                "qw": qw,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "f": c,
                "cx": cx,
                "cy": cy,
            }

    return mapping


# Strraightforward, load all data streams, generate record file 
def main(orient_path, images_dir, depths_dir, out_csv, intrinsics=None, skip_missing=False):

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    orient = parse_orientations(orient_path)

    rgb_files = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    rows = []

    for rgb_name in rgb_files:
        rgb_path = os.path.join(images_dir, rgb_name)

        # Orientation key: remove "_res"
        orient_key = rgb_name.replace("_res", "")

        if orient_key not in orient:
            msg = f"[WARN] Missing orientation for {rgb_name} (key {orient_key})"
            print(msg)
            if skip_missing:
                continue
            else:
                raise RuntimeError(msg)

        # Depth filename
        depth_name = rgb_name.replace("_res.jpg", "_depth_res.npy").replace("_res.jpeg", "_depth_res.npy")
        depth_path = os.path.join(depths_dir, depth_name)

        if not os.path.isfile(depth_path):
            msg = f"[WARN] Missing depth for {rgb_name}: {depth_path}"
            print(msg)
            if skip_missing:
                continue
            else:
                raise RuntimeError(msg)

        o = orient[orient_key]

        if intrinsics is not None:
            fx, fy, cx, cy = intrinsics
        else:
            fx = fy = o["f"]
            cx = o["cx"]
            cy = o["cy"]

        rows.append({
            "RGB_im": os.path.normpath(rgb_path),
            "depth": os.path.normpath(depth_path),
            "f_x": fx,
            "f_y": fy,
            "c_x": cx,
            "c_y": cy,
            "rot_w": o["qw"],
            "rot_x": o["qx"],
            "rot_y": o["qy"],
            "rot_z": o["qz"],
            "trans_x": o["X0"],
            "trans_y": o["Y0"],
            "trans_z": o["Z0"],
        })


    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")

        writer.writerow([
            "RGB_im", "depth",
            "f_x", "f_y", "c_x", "c_y",
            "rot_w", "rot_x", "rot_y", "rot_z",
            "trans_x", "trans_y", "trans_z",
        ])

        for r in rows:

            # sanitize translation fields
            tx_list = sanitize_numeric_field(f"{r['trans_x']:.9f}")
            ty_list = sanitize_numeric_field(f"{r['trans_y']:.9f}")
            tz_list = sanitize_numeric_field(f"{r['trans_z']:.9f}")

            if len(tx_list) > 1 or len(ty_list) > 1 or len(tz_list) > 1:
                print("   trans_x:", tx_list)
                print("   trans_y:", ty_list)
                print("   trans_z:", tz_list)

            tx = tx_list[0]
            ty = ty_list[0]
            tz = tz_list[0]

            writer.writerow([
                r["RGB_im"],
                r["depth"],
                f"{r['f_x']:.6f}",
                f"{r['f_y']:.6f}",
                f"{r['c_x']:.6f}",
                f"{r['c_y']:.6f}",
                f"{r['rot_w']:.16f}",
                f"{r['rot_x']:.16f}",
                f"{r['rot_y']:.16f}",
                f"{r['rot_z']:.16f}",
                tx,
                ty,
                tz,
            ])

    print(f"[OK] Wrote {len(rows)} rows to {out_csv}")


# each option here represents a lesson I learned the hard way, especially --skip-missing
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--orient", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--depths", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--skip-missing", action="store_true")
    args = ap.parse_args()

    intr = None
    if args.fx is not None:
        intr = (args.fx, args.fy, args.cx, args.cy)

    main(args.orient, args.images, args.depths, args.out, intrinsics=intr, skip_missing=args.skip_missing)

