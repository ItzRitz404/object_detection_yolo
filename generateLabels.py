import json
from pathlib import Path

base = Path("/home/ritz/Documents/object_detection")
root = base / "raw/labels/mtsd_v2_fully_annotated"
ann_dir = root / "annotations"
split_dir = root / "splits"

out_train = base / "dataset/labels/train"
out_val = base / "dataset/labels/val"
out_train.mkdir(parents=True, exist_ok=True)
out_val.mkdir(parents=True, exist_ok=True)

train_ids = {x.strip() for x in (split_dir / "train.txt").read_text().splitlines() if x.strip()}
val_ids = {x.strip() for x in (split_dir / "val.txt").read_text().splitlines() if x.strip()}

# collect all class names automatically
class_names = set()
for p in ann_dir.glob("*.json"):
    data = json.loads(p.read_text())
    for obj in data.get("objects", []):
        label = obj.get("label")
        if label:
            class_names.add(label)

class_names = sorted(class_names)
class_map = {name: i for i, name in enumerate(class_names)}

# save class list
(base / "dataset/classes.txt").write_text("\n".join(class_names) + "\n")

def to_yolo(xmin, ymin, xmax, ymax, w, h):
    xc = ((xmin + xmax) / 2.0) / w
    yc = ((ymin + ymax) / 2.0) / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return xc, yc, bw, bh

made_train = 0
made_val = 0

for p in ann_dir.glob("*.json"):
    image_id = p.stem
    data = json.loads(p.read_text())

    width = data["width"]
    height = data["height"]
    objects = data.get("objects", [])

    lines = []
    for obj in objects:
        label = obj.get("label")
        bbox = obj.get("bbox", {})

        xmin = bbox.get("xmin")
        ymin = bbox.get("ymin")
        xmax = bbox.get("xmax")
        ymax = bbox.get("ymax")

        if None in (label, xmin, ymin, xmax, ymax):
            continue

        cls_id = class_map[label]
        xc, yc, bw, bh = to_yolo(float(xmin), float(ymin), float(xmax), float(ymax), float(width), float(height))
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    if image_id in train_ids:
        out_file = out_train / f"{image_id}.txt"
        made_train += 1
    elif image_id in val_ids:
        out_file = out_val / f"{image_id}.txt"
        made_val += 1
    else:
        continue

    out_file.write_text("\n".join(lines) + ("\n" if lines else ""))

print("train labels:", made_train)
print("val labels:", made_val)
print("classes:", len(class_names))
print("saved class list to:", base / "dataset/classes.txt")
