import argparse
import json
import os
import time

HOME = os.getenv("HOME")

img_path = "{root}/{dataset}/{split}/images"
label_path = "{root}/{dataset}/{split}/labels"
annotation_file = "{dataset}_v1_{split}.json"

def write_annotation(annotation: dict, label_path: str):
    img_id = annotation["image_id"]
    filename = f"{str(img_id).zfill(12)}.txt"

    label = annotation["category_id"]
    bounding_box = annotation["bbox"] # LVIS bbox coordinate is "xywh" format
    output = str(label) + " " + " ".join([str(x) for x in bounding_box]) + "\n"

    with open(os.path.join(label_path, filename), "a") as f:
        f.write(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="lvis",
        required=False,
        help="Dataset name"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "val", "test"),
        required=True,
        help="Dataset split to be processed."
    )
    args = parser.parse_args()

    img_path = img_path.format(
        root=HOME,
        dataset=args.dataset,
        split=args.split,
    )
    label_path = label_path.format(
        root=HOME,
        dataset=args.dataset,
        split=args.split,
    )
    annotation_file = annotation_file.format(
        dataset=args.dataset,
        split=args.split,
    )
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"{img_path} doens't exist yet, run download.sh to fetch and unzip data.")

    if not os.path.exists(label_path):
        os.makedirs(label_path)

    metadata = json.load(open(f"{HOME}/{annotation_file}", "r"))
    annotations = metadata["annotations"]
    
    # stats = {m["id"]: (m["width"], m["height"]) for m in metadata["images"]}

    # ~40s to write ~1 million annotations
    # Not too slow, but could be improved by async?
    start = time.perf_counter()
    for a in annotations:
        # id = a["image_id"]
        # image_w, image_h = stats[id]

        # Convert (x, y) top-left box coordinate to box centre
        x, y, w, h = a["bbox"]
        a["bbox"] = [x + w / 2, y + h / 2, w, h]
        write_annotation(a, label_path)
    
    elapsed = time.perf_counter() - start
    print(f"Processed {len(annotations)} annotations in {round(elapsed, 2)} seconds.")