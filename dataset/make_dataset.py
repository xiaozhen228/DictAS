import sys
import shutil
import os
from datasets.VisA import Visa_dataset
from datasets.MVTec import Mvtec_dataset
from datasets.BTAD import BTAD_dataset
from datasets.MPDD import MPDD_dataset
from datasets.MVTec3D import Mvtec3D_dataset
from datasets.RESC import RESC_dataset
from datasets.BrasTS import BrasTS_dataset
from datasets.VOC import Ade_dataset


def move(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def process_dataset(dataset_cls, src_root, des_root, id_start=0, binary=True, to_255=True):

    move(des_root)
    dataset = dataset_cls(src_root)
    return dataset.make_VAND(binary=binary, to_255=to_255, des_path_root=des_root, id=id_start)


if __name__ == "__main__":
    id_counter = 0


    datasets_config = [
        {
            "name": "visa",   # https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
            "class": Visa_dataset,
            "src": "root to visa/visa",
            "des": "./dataset/mvisa/data/visa"
        },
        {
            "name": "mvtec",  # https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads
            "class": Mvtec_dataset,
            "src": "root to MVTec-AD/MVtec_ad",
            "des": "./dataset/mvisa/data/mvtec"
        },
        {
            "name": "BTAD",  # Please download our DATA_Google.zip:  https://drive.google.com/file/d/1DDFIquy_rcfcgqIymYIY76kBTXmeLpOj/view?usp=drive_link
            "class": BTAD_dataset,
            "src": "root to BTAD/BTech_Dataset_transformed",
            "des": "./dataset/mvisa/data/BTAD"
        },
        {
            "name": "MPDD", # https://github.com/stepanje/MPDD
            "class": MPDD_dataset,
            "src": "root to MPDD/MPDD",
            "des": "./dataset/mvisa/data/MPDD"
        },
        {
            "name": "mvtec3D",   # https://www.mvtec.com/company/research/datasets/mvtec-3d-ad/downloads
            "class": Mvtec3D_dataset,
            "src": "root to mvtec3D/MVTec_3D",
            "des": "./dataset/mvisa/data/mvtec3D"
        },
        {
            "name": "RESC",   # Please download our DATA_Google.zip:  https://drive.google.com/file/d/1DDFIquy_rcfcgqIymYIY76kBTXmeLpOj/view?usp=drive_link
            "class": RESC_dataset,
            "src": "root to RESC/RESC",
            "des": "./dataset/mvisa/data/RESC"
        },
        {
            "name": "BrasTS",   # Please download our DATA_Google.zip:  https://drive.google.com/file/d/1DDFIquy_rcfcgqIymYIY76kBTXmeLpOj/view?usp=drive_link
            "class": BrasTS_dataset,
            "src": "root to BrasTS/BrasTS",
            "des": "./dataset/mvisa/data/BrasTS"
        },
        {
            "name": "Ade20K",   # Please download our DATA_Google.zip:  https://drive.google.com/file/d/1DDFIquy_rcfcgqIymYIY76kBTXmeLpOj/view?usp=drive_link
            "class": Ade_dataset,
            "src": "root to Ade20K/Ade",
            "des": "./dataset/mvisa/data/Ade"
        },
        {
            "name": "VOC",   # Please download our DATA_Google.zip:  https://drive.google.com/file/d/1DDFIquy_rcfcgqIymYIY76kBTXmeLpOj/view?usp=drive_link
            "class": Ade_dataset,
            "src": "root to VOC2012/VOC",
            "des": "./dataset/mvisa/data/VOC"
        },

    ]

    for config in datasets_config:
        print(f"Processing {config['name']}...")
        id_counter = process_dataset(
            dataset_cls=config["class"],
            src_root=config["src"],
            des_root=config["des"],
            id_start= 0
        )
        print(f"Finished {config['name']}, next ID: {id_counter}")

    