import os
import glob
import time
import psutil
import open3d as o3d
import numpy as np
from tqdm import tqdm

raw_data_dir_root = "data/raw/ModelNet40"
output_dir_root = "data/processed"

NUM_POINTS = 2048

categories = ["airplane", "car", "chair", "sofa", "table"]
splits = ["train", "test"]

MAX_FILES_PER_SPLIT = 100

def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_bytes = process.memory_info().rss
    return memory_bytes/ (1024 ** 2)

def off_to_pointcloud_convert(off_path, num_points = 2048):
    mesh = o3d.io.read_triangle_mesh(off_path)

    if not mesh.has_vertices():
        raise ValueError(f"{off_path} --> mesh has no vertices")
    
    mesh.compute_triangle_normals()
    pointcloud = mesh.sample_points_poisson_disk(num_points)
    matrix = np.asarray(pointcloud.points)

    return matrix

def main():

    start_time = time.time()
    total_files = 0
    total_memo_saved = 0

    os.makedirs(output_dir_root, exist_ok =  True)

    for category in categories:
        for split in splits:
            source_dir = os.path.join(raw_data_dir_root, category, split)
            if not os.path.isdir(source_dir):
                print(f"{source_dir} missing --> skipping")
                continue

            output_dir = os.path.join(output_dir_root, category, split)
            os.makedirs(output_dir, exist_ok = True)

            off_files = glob.glob(os.path.join(source_dir, "*.off"))
            if not off_files:
                print(f"no .off files????")
                continue

            off_files = off_files[:MAX_FILES_PER_SPLIT]

            print(f"\n processing {category}/{split} ({len(off_files)} files)")

            for off_path in tqdm(off_files, desc = f" converting", unit = "file"):
                base_name = os.path.basename(off_path).replace(".off", "txt")
                txt_path = os.path.join(output_dir, base_name)

                if os.path.exists(txt_path):
                    total_files += 1
                    continue

                memory_before = get_memory_usage()

                try:
                    points = off_to_pointcloud_convert(off_path, NUM_POINTS)
                    np.savetxt(txt_path, points, fmt = "%.6f", delimiter = " ")
                    total_files += 1

                    memory_after = get_memory_usage()
                    memory_temp = memory_after - memory_before
                    total_memo_saved += memory_temp

                except Exception as e:
                    print(f"error processing of  {off_path}: {e}")

    elapsed = time.time() - start_time
    print(f"files converted : {total_files}")
    print(f"total time      : {elapsed:.2f} seconds")
    print(f"total memory used (approx) : {total_memo_saved:.1f} MB")
    print("output folder   :", os.path.abspath(output_dir_root))

if __name__ == "__main__":
    main()