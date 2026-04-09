from pathlib import Path

base = Path("/home/ritz/Documents/object_detection/dataset")
classes = [x.strip() for x in (base / "classes.txt").read_text().splitlines() if x.strip()]

lines = [
    f"path: {base}",
    "train: images/train",
    "val: images/val",
    "",
    "names:"
]

for i, name in enumerate(classes):
    lines.append(f"  {i}: {name}")

(base / "data.yaml").write_text("\n".join(lines) + "\n")
print("saved:", base / "data.yaml")
print()
print((base / "data.yaml").read_text()[:1500])