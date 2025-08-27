import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import open3d as o3d
from pathlib import Path

modelnet40_label_to_idx = {
    "airplane": 0,
    "bathtub": 1,
    "bed": 2,
    "bench": 3,
    "bookshelf": 4,
    "bottle": 5,
    "bowl": 6,
    "car": 7,
    "chair": 8,
    "cone": 9,
    "cup": 10,
    "curtain": 11,
    "desk": 12,
    "door": 13,
    "dresser": 14,
    "flower_pot": 15,
    "glass_box": 16,
    "guitar": 17,
    "keyboard": 18,
    "lamp": 19,
    "laptop": 20,
    "mantel": 21,
    "monitor": 22,
    "night_stand": 23,
    "person": 24,
    "piano": 25,
    "plant": 26,
    "radio": 27,
    "range_hood": 28,
    "sink": 29,
    "sofa": 30,
    "stairs": 31,
    "stool": 32,
    "table": 33,
    "tent": 34,
    "toilet": 35,
    "tv_stand": 36,
    "vase": 37,
    "wardrobe": 38,
    "xbox": 39
}

# the open3d mesh loader can't parse OFF if it starts with "OFF1234" etc.
# move the integers one line below
def fix_off_files(files_dir):
    for root, _, files in os.walk(files_dir):
        for file in files:
            if not file.endswith(".off"): continue


            path = os.path.join(root, file)
            print(f"fixing file {path}")
            with open(path, "r") as f:
                lines = f.readlines()

            if not lines:
                continue

            first_line = lines[0].strip()

            # Case 1: Already correct (first line exactly "OFF")
            if first_line == "OFF":
                continue

            print("moving values one line below OFF")

            # Case 2: Broken case (like "OFF1568 1820 0")
            if first_line.startswith("OFF") and len(first_line) > 3:
                # Split into "OFF" and rest of line
                rest = first_line[3:].strip()

                # Rewrite file with fixed header
                new_lines = ["OFF\n"]
                if rest:  # only add if something remains
                    new_lines.append(rest + "\n")
                new_lines.extend(lines[1:])

                with open(path, "w") as f:
                    f.writelines(new_lines)
                print(f"Fixed: {path}")

def plot_voxel_grid(vg):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, max(vg.shape))
    ax.set_ylim(0, max(vg.shape))
    ax.set_zlim(0, max(vg.shape))
    vg_bool = vg.astype(bool)
    ax.voxels(vg_bool, facecolors='red', edgecolor='k')
    ax.view_init(30, 45)
    plt.show()

# calculates the smallest possible voxel grid we need as the model input by loading every mesh.
# repeated code, but otherwise it takes a lot of resources to process the meshes.
def get_model_grid_shape(meshes_dir, voxel_size):
    model_grid_shape = [0, 0, 0]
    for root, _, files in os.walk(meshes_dir):
        for file in files:
            if not file.endswith(".off"): continue
            print(f"loading file: {os.path.join(root, file)}")
            mesh = o3d.io.read_triangle_mesh(os.path.join(root, file))
            mesh.compute_vertex_normals()

            mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
                       center=mesh.get_center())

            voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)

            voxels = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])

            max_idx_0 = max(x[0] for x in voxels)
            max_idx_1 = max(x[1] for x in voxels)
            max_idx_2 = max(x[2] for x in voxels)
            grid_shape = (max_idx_0 + 1, max_idx_1 + 1, max_idx_2 + 1)
            print(f"calculating grid shape of {file}")
            print(f"grid shape: {grid_shape}")

            model_grid_shape[0] = grid_shape[0] if grid_shape[0] > model_grid_shape[0] else model_grid_shape[0]
            model_grid_shape[1] = grid_shape[1] if grid_shape[1] > model_grid_shape[1] else model_grid_shape[1]
            model_grid_shape[2] = grid_shape[2] if grid_shape[2] > model_grid_shape[2] else model_grid_shape[2]
            print(f"current model grid shape: {model_grid_shape}")

    return model_grid_shape

def process_meshes(meshes_dir, output_dir, voxel_size):

    model_grid_shape = get_model_grid_shape(meshes_dir, voxel_size)
    print("#########################################################")
    print(f"MODEL GRID SHAPE: {model_grid_shape}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.savetxt(os.path.join(output_dir, "model_grid_shape.txt"), model_grid_shape)
    print("#########################################################")

    for root, _, files in os.walk(meshes_dir):
        for file in files:
            if not file.endswith(".off"): continue

            print(f"Processing {root}\\{file}")

            out_filename = file[:-4]
            out_out_dir = root[len(meshes_dir):]
            if Path(os.path.join(output_dir + out_out_dir, f"{out_filename}.npy")).exists():
                print(f"File {os.path.join(output_dir + out_out_dir, f"{out_filename}.npy")} already exists, skipping")
                continue

            mesh = o3d.io.read_triangle_mesh(os.path.join(root, file))
            mesh.compute_vertex_normals()

            mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())

            voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)

            voxels = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
            print(f"voxels: {voxels}")
            print(f"number of voxels: {len(voxels)}")

            # # Normalize indices to start at (0,0,0)
            # min_idx_0 = min(x[0] for x in voxels)
            # min_idx_1 = min(x[1] for x in voxels)
            # min_idx_2 = min(x[2] for x in voxels)
            # voxels = np.subtract(voxels, (min_idx_0, min_idx_1, min_idx_2))

            max_idx_0 = max(x[0] for x in voxels)
            max_idx_1 = max(x[1] for x in voxels)
            max_idx_2 = max(x[2] for x in voxels)
            grid_shape = (max_idx_0 + 1, max_idx_1 + 1, max_idx_2 + 1)
            print(f"grid_shape: {grid_shape} (model grid shape: {model_grid_shape})")

            # model grid will be our input to the model, it has a fixed shape
            model_grid = np.zeros(model_grid_shape)
            for v in voxels:
                model_grid[v[0], v[1], v[2]] = 1

            # plot_voxel_grid(model_grid)

            Path(os.path.join(output_dir + out_out_dir)).mkdir(parents=True, exist_ok=True)
            np.save(os.path.join(output_dir + out_out_dir, f"{out_filename}.npy"), model_grid)
            print(f"-- file: {out_filename}")
            print(f"Output: {output_dir + out_out_dir}\\{out_filename}.npy")

    print("#########################################################")
    print(f"MODEL GRID SHAPE: {model_grid_shape}")
    print("#########################################################")

def get_label_id(label_str):
    return modelnet40_label_to_idx[label_str]

def get_label_str(filename):
    l = filename.split("_")
    s = ""
    for i in range(len(l) - 1):
        s = s + l[i]
        if i != len(l) - 2: s = s + "_"
    return s

def label_id_to_np(label_id):
    t = np.zeros(len(modelnet40_label_to_idx))
    t[label_id] = 1
    return t

# type either "train" or "test"
class VGDataset(Dataset):

    def __init__(self, dataset_path, type: str):
        self.data = []
        self.labels = []
        self.label_set = {}
        self.path = dataset_path

        for root, _, files in os.walk(self.path):
            for file in files:
                if not file.endswith(".npy"): continue

                parent_dir_name = os.path.basename(root)
                if parent_dir_name != type: continue

                d = torch.from_numpy(np.load(os.path.join(root, file), allow_pickle=True)).float()
                self.data.append(d)
                self.labels.append(torch.from_numpy(label_id_to_np(get_label_id(get_label_str(file)))))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == "__main__":
    # process_meshes("./data/dummy/", "./data/dummy/out/", VOXEL_SIZE)
    DATASET_PATH = "./data/ModelNet40/"
    DATASET_PROCESSED_PATH = "./data/out/"
    VOXEL_SIZE = 0.02
    process_meshes(DATASET_PATH, DATASET_PROCESSED_PATH, VOXEL_SIZE)
    # fix_off_files(DATASET_PATH)
