import os
import glob as file_glob
from PIL import Image

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "appended_cases")
os.makedirs(output_dir, exist_ok=True)

case_dirs = sorted([
    d for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d))
    and not d.startswith(".")
    and d != "appended_cases"
])

VIEW_ORDER = ["axial", "coronal", "sagittal"]

for case_name in case_dirs:
    case_path = os.path.join(base_dir, case_name)
    append_path = os.path.join(case_path, "append")

    if not os.path.isdir(append_path):
        print(f"Skipping {case_name}: no append folder")
        continue

    pngs = sorted(file_glob.glob(os.path.join(append_path, "*.png")))

    ordered = []
    for view in VIEW_ORDER:
        matches = [p for p in pngs if f"_{view}_slice" in os.path.basename(p)]
        if matches:
            ordered.append(matches[0])
        else:
            print(f"WARNING: {case_name} missing {view} image")

    if len(ordered) < 3:
        print(f"Skipping {case_name}: not enough images ({len(ordered)})")
        continue

    images = [Image.open(p) for p in ordered]
    widths = [img.width for img in images]
    heights = [img.height for img in images]

    max_width = max(widths)
    total_height = sum(heights)

    combined = Image.new("RGB", (max_width, total_height))

    y = 0
    for img in images:
        w = img.width
        x_offset = (max_width - w) // 2
        combined.paste(img, (x_offset, y))
        y += img.height

    out_path = os.path.join(output_dir, f"{case_name}_appended.png")
    combined.save(out_path)
    print(f"Saved: {case_name}_appended.png  ({max_width}x{total_height})")

print(f"\nDone. Outputs in: {output_dir}")
