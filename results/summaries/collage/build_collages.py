import os
import re
import subprocess
import sys
import tempfile
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

VIS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "Visualization")
OUT_DIR = os.path.join(os.path.dirname(__file__))

SUBDIRS = ["10-CASES", "5 Cases"]
VIEWS = ["axial", "coronal", "sagittal"]


def natural_key(filename):
    basename = os.path.basename(filename)
    m = re.search(r"slice(\d+\.?\d*)", basename)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return float(m.group(1))
    digits = re.findall(r"\d+", basename)
    return int(digits[-1]) if digits else 0


def build_strip(files, label):
    """Vertically stack a list of images into one strip using PIL."""
    if not files:
        return None

    first = Image.open(files[0])
    width = first.size[0]
    slice_h = first.size[1]
    mode = first.mode
    first.close()

    total_h = len(files) * slice_h
    strip = Image.new(mode, (width, total_h))

    for i, f in enumerate(files):
        with Image.open(f) as img:
            strip.paste(img, (0, i * slice_h))
        if (i + 1) % 100 == 0:
            print(f"    {label} {i+1}/{len(files)}")

    return strip


def process_case(case_dir, case_name):
    """Create a vertically stacked collage using PIL + ImageMagick hybrid."""
    output_path = os.path.join(OUT_DIR, f"{case_name}.png")

    if os.path.exists(output_path):
        print(f"  SKIP: {output_path} already exists")
        return True

    slice_counts = {}
    view_files = {}

    for view in VIEWS:
        view_dir = os.path.join(case_dir, view)
        if not os.path.isdir(view_dir):
            print(f"  WARNING: {view_dir} not found")
            continue

        png_files = [
            os.path.join(view_dir, f)
            for f in os.listdir(view_dir)
            if f.lower().endswith(".png")
        ]
        png_files.sort(key=natural_key)
        slice_counts[view] = len(png_files)
        view_files[view] = png_files

    total = sum(slice_counts.values())
    print(f"  {case_name}: {slice_counts} -> {total} total slices")

    if total == 0:
        print(f"  ERROR: no images found in {case_dir}")
        return False

    strip_paths = []
    try:
        for view in VIEWS:
            files = view_files.get(view, [])
            if not files:
                continue
            print(f"    Building {view} strip ({len(files)} slices)...")
            strip = build_strip(files, view)
            if strip is None:
                continue
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            strip.save(tmp.name, "PNG")
            strip.close()
            strip_paths.append(tmp.name)
            print(f"    {view} strip saved ({len(files)} slices)")

        if len(strip_paths) < 2:
            print(f"  ERROR: not enough strips to combine")
            return False

        print(f"    Combining {len(strip_paths)} strips...")
        cmd = ["magick"] + strip_paths + ["-append", output_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    ImageMagick error: {result.stderr.strip()}")
            return False

        size_kb = os.path.getsize(output_path) / 1024
        print(f"  DONE: {size_kb:.0f} KB")
        return True

    finally:
        for p in strip_paths:
            try:
                os.unlink(p)
            except OSError:
                pass


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_cases = []
    for subdir in SUBDIRS:
        subdir_path = os.path.join(VIS_DIR, subdir)
        if not os.path.isdir(subdir_path):
            continue

        for entry in sorted(os.listdir(subdir_path)):
            case_path = os.path.join(subdir_path, entry)
            if os.path.isdir(case_path) and not entry.startswith("."):
                all_cases.append((case_path, entry))

    if not all_cases:
        print("ERROR: No case folders found!")
        sys.exit(1)

    print(f"Found {len(all_cases)} cases to process\n")

    success = 0
    fail = 0
    for case_path, case_name in all_cases:
        print(f"Processing: {case_name}")
        if process_case(case_path, case_name):
            success += 1
        else:
            fail += 1
            print(f"  FAILED: {case_name}")

    print(f"\n--- Summary ---")
    print(f"Success: {success}, Failed: {fail}")


if __name__ == "__main__":
    main()
