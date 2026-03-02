import os
import re
import glob

"""
adjusts csv/png pair file index names so that no gaps remain
"""


# Folder with the PNG/CSV files
FOLDER = "data/"

# Match frame_00000_gray.png or frame_00000_depth.csv
pattern = re.compile(r"frame_(\d{5})_(gray|depth)\.(png|csv)")

# Gather all files
files = sorted(glob.glob(os.path.join(FOLDER, "frame_*_*.*")))

entries = {}

# --- Step 1: Parse valid filenames ---
for f in files:
    name = os.path.basename(f)
    m = pattern.match(name)
    if not m:
        continue

    idx, kind, ext = m.groups()
    idx = int(idx)

    if idx not in entries:
        entries[idx] = {}

    entries[idx][kind] = f  # store file path for this index & type

# Sort by original index
sorted_indices = sorted(entries.keys())

print(f"Found {len(sorted_indices)} valid frame indices.")

# --- Step 2: Rename them sequentially starting from 0 ---
for new_idx, old_idx in enumerate(sorted_indices):
    new_idx_str = f"{new_idx:05d}"

    entry = entries[old_idx]

    # rename gray frame if exists
    if "gray" in entry:
        old_path = entry["gray"]
        new_name = f"frame_{new_idx_str}_gray.png"
        new_path = os.path.join(FOLDER, new_name)
        os.rename(old_path, new_path)
        print(f"{os.path.basename(old_path)} → {new_name}")

    # rename depth frame if exists
    if "depth" in entry:
        old_path = entry["depth"]
        new_name = f"frame_{new_idx_str}_depth.csv"
        new_path = os.path.join(FOLDER, new_name)
        os.rename(old_path, new_path)
        print(f"{os.path.basename(old_path)} → {new_name}")

print("\ndone, all frames renumbered without gaps.")
