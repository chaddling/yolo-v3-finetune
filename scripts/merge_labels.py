import argparse
import json
import logging
import os
import pandas as pd

from typing import List, Tuple

logging.getLogger().setLevel(logging.INFO)

HOME = os.getenv("HOME")

# This mapping is extracted from the official LVIS mapping:
# https://github.com/lvis-dataset/lvis-api/blob/master/data/coco_to_synset.json
COCO_SYNSET_CATEGORIES = [
    {"synset": "person.n.01", "coco_cat_id": 1},
    {"synset": "bicycle.n.01", "coco_cat_id": 2},
    {"synset": "car.n.01", "coco_cat_id": 3},
    {"synset": "motorcycle.n.01", "coco_cat_id": 4},
    {"synset": "airplane.n.01", "coco_cat_id": 5},
    {"synset": "bus.n.01", "coco_cat_id": 6},
    {"synset": "train.n.01", "coco_cat_id": 7},
    {"synset": "truck.n.01", "coco_cat_id": 8},
    {"synset": "boat.n.01", "coco_cat_id": 9},
    {"synset": "traffic_light.n.01", "coco_cat_id": 10},
    {"synset": "fireplug.n.01", "coco_cat_id": 11},
    {"synset": "stop_sign.n.01", "coco_cat_id": 13},
    {"synset": "parking_meter.n.01", "coco_cat_id": 14},
    {"synset": "bench.n.01", "coco_cat_id": 15},
    {"synset": "bird.n.01", "coco_cat_id": 16},
    {"synset": "cat.n.01", "coco_cat_id": 17},
    {"synset": "dog.n.01", "coco_cat_id": 18},
    {"synset": "horse.n.01", "coco_cat_id": 19},
    {"synset": "sheep.n.01", "coco_cat_id": 20},
    {"synset": "beef.n.01", "coco_cat_id": 21},
    {"synset": "elephant.n.01", "coco_cat_id": 22},
    {"synset": "bear.n.01", "coco_cat_id": 23},
    {"synset": "zebra.n.01", "coco_cat_id": 24},
    {"synset": "giraffe.n.01", "coco_cat_id": 25},
    {"synset": "backpack.n.01", "coco_cat_id": 27},
    {"synset": "umbrella.n.01", "coco_cat_id": 28},
    {"synset": "bag.n.04", "coco_cat_id": 31},
    {"synset": "necktie.n.01", "coco_cat_id": 32},
    {"synset": "bag.n.06", "coco_cat_id": 33},
    {"synset": "frisbee.n.01", "coco_cat_id": 34},
    {"synset": "ski.n.01", "coco_cat_id": 35},
    {"synset": "snowboard.n.01", "coco_cat_id": 36},
    {"synset": "ball.n.06", "coco_cat_id": 37},
    {"synset": "kite.n.03", "coco_cat_id": 38},
    {"synset": "baseball_bat.n.01", "coco_cat_id": 39},
    {"synset": "baseball_glove.n.01", "coco_cat_id": 40},
    {"synset": "skateboard.n.01", "coco_cat_id": 41},
    {"synset": "surfboard.n.01", "coco_cat_id": 42},
    {"synset": "tennis_racket.n.01", "coco_cat_id": 43},
    {"synset": "bottle.n.01", "coco_cat_id": 44},
    {"synset": "wineglass.n.01", "coco_cat_id": 46},
    {"synset": "cup.n.01", "coco_cat_id": 47},
    {"synset": "fork.n.01", "coco_cat_id": 48},
    {"synset": "knife.n.01", "coco_cat_id": 49},
    {"synset": "spoon.n.01", "coco_cat_id": 50},
    {"synset": "bowl.n.03", "coco_cat_id": 51},
    {"synset": "banana.n.02", "coco_cat_id": 52},
    {"synset": "apple.n.01", "coco_cat_id": 53},
    {"synset": "sandwich.n.01", "coco_cat_id": 54},
    {"synset": "orange.n.01", "coco_cat_id": 55},
    {"synset": "broccoli.n.01", "coco_cat_id": 56},
    {"synset": "carrot.n.01", "coco_cat_id": 57},
    {"synset": "frank.n.02", "coco_cat_id": 58},
    {"synset": "pizza.n.01", "coco_cat_id": 59},
    {"synset": "doughnut.n.02", "coco_cat_id": 60},
    {"synset": "cake.n.03", "coco_cat_id": 61},
    {"synset": "chair.n.01", "coco_cat_id": 62},
    {"synset": "sofa.n.01", "coco_cat_id": 63},
    {"synset": "pot.n.04", "coco_cat_id": 64},
    {"synset": "bed.n.01", "coco_cat_id": 65},
    {"synset": "dining_table.n.01", "coco_cat_id": 67},
    {"synset": "toilet.n.02", "coco_cat_id": 70},
    {"synset": "television_receiver.n.01", "coco_cat_id": 72},
    {"synset": "laptop.n.01", "coco_cat_id": 73},
    {"synset": "mouse.n.04", "coco_cat_id": 74},
    {"synset": "remote_control.n.01", "coco_cat_id": 75},
    {"synset": "computer_keyboard.n.01", "coco_cat_id": 76},
    {"synset": "cellular_telephone.n.01", "coco_cat_id": 77},
    {"synset": "microwave.n.02", "coco_cat_id": 78},
    {"synset": "oven.n.01", "coco_cat_id": 79},
    {"synset": "toaster.n.02", "coco_cat_id": 80},
    {"synset": "sink.n.01", "coco_cat_id": 81},
    {"synset": "electric_refrigerator.n.01", "coco_cat_id": 82},
    {"synset": "book.n.01", "coco_cat_id": 84},
    {"synset": "clock.n.01", "coco_cat_id": 85},
    {"synset": "vase.n.01", "coco_cat_id": 86},
    {"synset": "scissors.n.01", "coco_cat_id": 87},
    {"synset": "teddy.n.01", "coco_cat_id": 88},
    {"synset": "hand_blower.n.01", "coco_cat_id": 89},
    {"synset": "toothbrush.n.01", "coco_cat_id": 90},
]

def map_merged_idx(lvis_categories: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List]:
    """
    Returns two dataframes corresponding to the COCO/LVIS dataset, where mapping the original
    category ids to a new merged index. Since we use a model pretrained on COCO, their indices
    go first in the order and the rest, non-overlapping LVIS categories receive a new index.

    :param lvis_categories: The object category metadata from LVIS.
    :return: Dataframes that maps coco_id -> merged_id, and from lvis_id -> merged_id.
        Also returns any unmerged COCO synset from COCO_SYNSET_CATEGORIES.
    """
    unmerged_synset = []
    idx = 1
    coco_vocabs = []

    for c in COCO_SYNSET_CATEGORIES:
        synset = c["synset"]
        try:
            vocab = lvis_categories.loc[synset, "name"]
        except KeyError:
            unmerged_synset.append(c)
            logging.info(f"Unmerged COCO synset: {c}.")

        coco_vocabs.append(
            {
                "name": vocab, 
                "synset": synset, 
                "coco_id": int(c["coco_cat_id"]), 
                "merged_id": idx
            }
        )
        idx += 1

    coco_vocabs = pd.DataFrame.from_records(coco_vocabs).set_index("synset")
    logging.info(f"Found {len(unmerged_synset)} from COCO_SYNSET_CATEGORIES.")

    lvis_vocabs = []
    for synset, row in lvis_categories.iterrows():
        merged_id = idx
        is_coco = False
        if synset in coco_vocabs.index:
            merged_id = coco_vocabs.loc[synset].merged_id
            is_coco = True

        lvis_vocabs.append(
            {
                "name": row["name"],
                "synset": synset,
                "lvis_id": int(row["id"]),
                "merged_id": merged_id,
            }
        )

        if not is_coco:
            idx += 1

    lvis_vocabs = pd.DataFrame.from_records(lvis_vocabs).set_index("synset")
    return coco_vocabs, lvis_vocabs, unmerged_synset


def get_unmerged_coco_annotations(
    coco_annotations: pd.DataFrame, 
    coco_images: pd.DataFrame, 
    unmerged_synset: List
) -> pd.DataFrame:
    """
    Retrieves the annotations associated with the unmerged category synset.
    Joining annotations to images will add a "coco_url" column that uniquely 
    identifies an image.

    :param coco_annotations: Annotation metadata from the COCO dataset.
    :param coco_images: Images metadata from the COCO dataset.
    :param unmerged synset: Any categories from COCO_SYNSET_CATEGORIES that
        are not found in the LVIS dataset.
    :return: A merged dataframe containing annotations for the unmerged object category. 
    """
    unmerged_category_ids = [x["coco_cat_id"] for x in unmerged_synset]
    coco_annotations_unmerged = coco_annotations[coco_annotations.category_id.isin(unmerged_category_ids)]
    logging.info(f"Found {len(coco_annotations_unmerged)} COCO annotations related to unmerged synsets/ids.")

    coco_metadata_unmerged = pd.merge(coco_annotations_unmerged, coco_images, how="left", on=["image_id"])

    return coco_metadata_unmerged


def apply_new_category_idx(
    coco_vocabs: pd.DataFrame, 
    lvis_vocabs: pd.DataFrame,
    lvis_annotations: pd.DataFrame,
    lvis_images: pd.DataFrame,
) -> pd.DataFrame:
    """
    1. Maps the loaded LVIS annotation metadata (lvis_annotations)'s category id to the merged index.
    2. Adds extra annotations to the LVIS annotations for the unmerged COCO object, by matching the corresponding
    "coco_url".

    :param coco_vocabs: Dataframe contanining a map of coco_id -> merged_id.
    :param lvis_vocabs: Dataframe contaiing a map of lvis_id -> merged_id.
    :param lvis_annotations: Dataframe containing the loaded, unmodified LVIS dataset annotations.
    :param lvis_images: Dataframe containing the LVIS images metadata.
    :return: A dataframe of LVIS dataset annotations containing the merged index + extra annotations from COCO.
    """
    lvis_ids_map = {k: v for k, v in zip(lvis_vocabs.lvis_id.tolist(), lvis_vocabs.merged_id.tolist())}
    coco_ids_map = {k: v for k, v in zip(coco_vocabs.coco_id.tolist(), coco_vocabs.merged_id.tolist())}

    for _, a in lvis_annotations.iterrows():
        a["category_id"] = lvis_ids_map[a["category_id"]]

    new_records = []
    annotation_id = len(lvis_annotations)
    for _, c in coco_metadata_unmerged.iterrows():
        try:
            new_records.append(
                {
                    "id": annotation_id + 1,
                    "image_id": lvis_images.loc[c["coco_url"]].image_id,
                    "category_id": coco_ids_map[c["category_id"]],
                    "segmentation": c["segmentation"],
                    "area": c["area"],
                    "bbox": c["bbox"],
                }
            )
        except KeyError:
            logging.warning(f"Image with url: {c['coco_url']} not found.")

    new_annotations = pd.DataFrame.from_records(new_records)

    logging.info(f"Before new annotations: {len(lvis_annotations)}")
    merged_annotations = pd.concat([lvis_annotations, new_annotations], axis=0)
    logging.info(f"After new annotations: {len(merged_annotations)}")
    return merged_annotations


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

    lvis_train = os.path.join(HOME, f"{args.dataset}_v1_{args.split}.json")
    try:
        with open(lvis_train, "r") as f:
            lvis_metadata = json.load(f)
    except FileNotFoundError as e:
        raise(e)
    
    lvis_categories = lvis_metadata["categories"]
    lvis_categories = pd.DataFrame.from_records(lvis_categories)
    lvis_categories = lvis_categories.set_index("synset")

    coco_vocabs, lvis_vocabs, unmerged_synset = map_merged_idx(lvis_categories)
    merged_len = len(set(list(lvis_vocabs.index) + list(coco_vocabs.index)))
    logging.info(f"There are {merged_len} tokens after merge.")

    coco_train = json.load(
        open(os.path.join(HOME, "annotations/instances_train2017.json"), "r")
    )
    coco_val = json.load(
        open(os.path.join(HOME, "annotations/instances_val2017.json"), "r")
    )
    coco_images = pd.DataFrame.from_records(coco_train["images"] + coco_val["images"])
    coco_images = coco_images[["coco_url", "id"]].rename(columns={"id": "image_id"})
    coco_annotations = pd.DataFrame.from_records(coco_train["annotations"] + coco_val["annotations"])

    coco_metadata_unmerged = get_unmerged_coco_annotations(coco_annotations, coco_images, unmerged_synset)

    lvis_annotations = lvis_metadata["annotations"]
    lvis_annotations = pd.DataFrame.from_records(lvis_annotations)

    lvis_images = lvis_metadata["images"]
    lvis_images = pd.DataFrame.from_records(lvis_images).rename(columns={"id": "image_id"})[["image_id", "coco_url"]].set_index("coco_url")

    merged_annotations = apply_new_category_idx(
        coco_vocabs,
        lvis_vocabs,
        lvis_annotations,
        lvis_images,
    )

    lvis_metadata["annotations"] = [
        x.to_dict() for _, x in merged_annotations.iterrows()
    ]
    with open(os.path.join(HOME, f"merged_{args.dataset}_v1_{args.split}.json"), "w") as f:
        json.dump(lvis_metadata, f)