import os
from pathlib import Path

for root, _, files in os.walk("./data/out"):
    for file in files:
        if "rotated_z0" in file or "rotated_z2" in file or "rotated_z4" in file or "rotated_z6" in file:
            print(f"removing {os.path.join(root, file)}")
            os.remove(os.path.join(root, file))

            # out_path = os.path.join(root, file).replace("ModelNet40", "out_augmenteds_backup")
            # print(f"HOBA {root.replace("ModelNet40", "out_augmenteds_backup")}")
            # Path(root.replace("ModelNet40", "out_augmenteds_backup")).mkdir(parents=True, exist_ok=True)
            # print(f"olala {os.path.join(root, file)} ---> {out_path}")
            # input()
            # os.rename(os.path.join(root, file), out_path)
