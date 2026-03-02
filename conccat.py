import os
import shutil
from pathlib import Path

"""
combines png/csv data from multiple folders into a single folder with straight indexing
"""


root = Path(".")
output_dir = root / "data"
output_dir.mkdir(exist_ok=True)

pair_index = 0  # global frame counter

# traverse every subfolder
for dirpath, _, filenames in os.walk(root):
    dirpath = Path(dirpath)
    if dirpath == output_dir:
        continue  # skip the destination folder itself

    # build lookup dictionaries for gray and depth files
    gray_files = {}
    depth_files = {}

    for name in filenames:
        if name.endswith("_gray.png"):
            prefix = name.replace("_gray.png", "")
            gray_files[prefix] = dirpath / name
        elif name.endswith("_depth.csv"):
            prefix = name.replace("_depth.csv", "")
            depth_files[prefix] = dirpath / name

    # find matching frame numbers that exist in both sets
    for prefix in sorted(set(gray_files) & set(depth_files)):
        new_index = f"{pair_index:05d}"

        # move gray file
        new_gray_name = f"frame_{new_index}_gray.png"
        shutil.move(str(gray_files[prefix]), output_dir / new_gray_name)

        # move depth file
        new_depth_name = f"frame_{new_index}_depth.csv"
        shutil.move(str(depth_files[prefix]), output_dir / new_depth_name)

        pair_index += 1

print(f"Moved {pair_index} frame pairs into '{output_dir}/'")
