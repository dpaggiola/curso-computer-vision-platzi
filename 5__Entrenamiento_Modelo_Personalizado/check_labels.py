import os
from pathlib import Path

root = Path("Labelling de conos_escalera y arco")
for split in ["train","val","test"]:
    imgs = sorted((root / split / "images").glob("*.*"))
    labels = sorted((root / split / "labels").glob("*.txt"))
    print(f"Split: {split} -> images: {len(imgs)}, labels: {len(labels)}")
    # show a sample label if exists
    if labels:
        print("Sample label (first):", labels[0])
        print(open(labels[0]).read())