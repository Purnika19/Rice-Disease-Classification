import os

data_dir = "data"
classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

print(f"Dataset classes: {classes}")
for cls in classes:
    count = len(os.listdir(os.path.join(data_dir, cls)))
    print(f"{cls}: {count} images")
