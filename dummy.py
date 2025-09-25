import os

for root, _, files in os.walk("./data/ModelNet40"):
    for file in files:
        if "rotated_x" in file or "rotated_y" in file or "rotated_z" in file:
            print(f"removing {os.path.join(root, file)}")
            os.remove(os.path.join(root, file))