import os
import subprocess
from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent
SRC = BASE_DIR / "bert_aug"
DATA_FILE = BASE_DIR / "scripts" / "dataset.txt"   # your .label dataset path
OUT_DIR = BASE_DIR / "augmented"
TMP_DIR = OUT_DIR / "tmp"
SEED = 42

os.makedirs(TMP_DIR, exist_ok=True)

tsv_path = TMP_DIR / "train.tsv"
try:
    with open(DATA_FILE, "r", encoding="utf-8") as fin, open(tsv_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            label, text = line.split(" ", 1)
            fout.write(f"{label}\t{text.strip()}\n")
except UnicodeDecodeError:
    with open(DATA_FILE, "r", encoding="latin-1") as fin, open(tsv_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            label, text = line.split(" ", 1)
            fout.write(f"{label}\t{text.strip()}\n")


augmentations = {
    "eda": [f"{SRC}/eda.py", "--num_aug=1", "--alpha=0.1"],
    # "cbert": [f"{SRC}/cbert.py", "--num_train_epochs=3"],
    # "cgpt2": [f"{SRC}/cgpt2.py", "--num_train_epochs=5", "--top_p=0.9", "--temp=1.0"],
}

augmented_files = []

for name, cmd in augmentations.items():
    aug_out = TMP_DIR / f"{name}_aug.tsv"
    print(f"\n🚀 Running {name.upper()} augmentation...")
    subprocess.run(
        ["python", *cmd,
         "--input", str(tsv_path),
         "--output", str(aug_out),
         "--seed", str(SEED)],
        check=True
    )
    if aug_out.exists():
        augmented_files.append(aug_out)
        print(f"{name.upper()} output → {aug_out}")
    else:
        print(f"{name.upper()} did not produce output, skipping.")


# --- STEP 3: Combine all augmentations ---
combined_path = OUT_DIR / "augmented_train1.label"
os.makedirs(OUT_DIR, exist_ok=True)

with open(combined_path, "w", encoding="utf-8") as fout:
    # Write original
    with open(DATA_FILE, "r", encoding="utf-8", errors="ignore") as fin:
        fout.writelines(fin.readlines())

    # Write each augmentation
    for aug_path in augmented_files:
        with open(aug_path, "r", encoding="utf-8", errors="ignore") as fin:
            for line in fin:
                if not line.strip():
                    continue
                label, text = line.strip().split("\t", 1)
                fout.write(f"{label} {text}\n")

print(f"\nCombined dataset saved to: {combined_path}")
print(f"Total augmentations combined: {len(augmented_files)}")
