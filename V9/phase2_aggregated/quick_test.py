import os
import config

print("Testing annotation paths...")
for case in config.TRAIN_CASES[:2]:  # Test first 2
    path = config.get_annotation_path(case)
    exists = os.path.exists(path)
    print(f"  {case}")
    print(f"    Path: {path}")
    print(f"    Exists: {exists}")
    print()
