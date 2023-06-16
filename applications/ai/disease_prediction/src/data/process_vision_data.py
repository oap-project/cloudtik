#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#
# SPDX-License-Identifier: BSD-3-Clause

import os
import pandas as pd
import argparse
import shutil
from os import path
from PIL import Image, ImageDraw
import json


def classify_images(
        image_folder, annotation_file, output_dir):
    print("Create folders for classified images")

    output_images_dir = os.path.join(output_dir, "vision_images")
    benign = os.path.join(output_images_dir, "Benign")
    malignant = os.path.join(output_images_dir, "Malignant")
    normal = os.path.join(output_images_dir, "Normal")
    os.makedirs(benign, exist_ok=True)
    os.makedirs(malignant, exist_ok=True)
    os.makedirs(normal, exist_ok=True)
    
    print("----- Classifying the images for data preprocessing -----")
    manual_annotations = pd.read_excel(annotation_file)
    print("Classify Low energy images")
    directory = os.path.join(image_folder, "Low energy images of CDD-CESM")
    for filename in os.listdir(directory):
        src = os.path.join(directory, filename)
        # checking if it is a file   
        if os.path.isfile(src):
            patient_id = filename.strip(".jpg")
            classification_type = manual_annotations[
                manual_annotations['Image_name'] == patient_id]["Pathology Classification/ Follow up"]
            if classification_type.size != 0:
                tgt = os.path.join(output_images_dir, str(classification_type.values[0]))
                shutil.copy2(src, tgt)             
    print("Classify Subtracted energy images")
    directory = os.path.join(image_folder, "Subtracted images of CDD-CESM")
    for filename in os.listdir(directory):
        src = os.path.join(directory, filename)
        # checking if it is a file   
        if os.path.isfile(src):
            patient_id = filename.strip(".jpg")
            classification_type = manual_annotations[
                manual_annotations['Image_name'] == patient_id]["Pathology Classification/ Follow up"]
            if classification_type.size != 0:
                tgt = os.path.join(output_images_dir, str(classification_type.values[0]))
                shutil.copy2(src, tgt)
    

def segment_images(segmentation_path, output_dir, cesm_only=True):
    new_width = 512
    new_height = 512

    df = pd.read_csv(segmentation_path)

    # iterate over files in
    # Creating segmented images for Normal cases
    output_images_dir = os.path.join(output_dir, "vision_images")
    output_segmented_dir = os.path.join(output_dir, "segmented_images")

    directory = path.join(output_images_dir, "Normal")
    save_directory = path.join(output_segmented_dir, "Normal")
    print(save_directory)
    os.makedirs(save_directory, exist_ok=True)
    print("Creating segmented images for \"Normal\" cases .........")
    for filename in os.listdir(directory):
        if cesm_only:
            if '_CM_' not in filename:
                continue
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            new_file = os.path.join(save_directory, filename)
            im = Image.open(f)
            width, height = im.size   # Get dimensions
            
            cx, cy = width//2, height//2
            left, top = max(0, cx - new_width//2), max(0, cy - new_height//2) 
            right, bottom = min(width, cx + new_width//2), min(height, cy + new_height//2)

            # Crop the center of the image
            im = im.crop((left, top, right, bottom))
            im.save(new_file)

    # Creating segmented images for malignant cases 
    directory = path.join(output_images_dir, "Malignant")
    save_directory = path.join(output_segmented_dir, "Malignant")
    os.makedirs(save_directory, exist_ok=True)
    print("Creating segmented images for \"malignant\" cases ..........")
    for filename in os.listdir(directory):
        if cesm_only:
            if '_CM_' not in filename:
                continue
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            rows = df.loc[df['#filename'] == filename]
            shapes = rows['region_shape_attributes'].values
            i = 0
            for shape in shapes:
                x = json.loads(shape)
                i = i + 1
                new_file = os.path.join(
                    save_directory,  filename.split('.')[0] + str(i) + '.' + filename.split('.')[1])
                im = Image.open(f)
                width, height = im.size   # Get dimensions    
                if x["name"] == "polygon":
                    left = min(x["all_points_x"])
                    top = min(x["all_points_y"])
                    right = max(x["all_points_x"])
                    bottom = max(x["all_points_y"])
                    """
                    if (right - left) < new_width:
                        left, right = max(0, (right+left)//2 - new_width//2), min(width,  (right+left)//2 + new_width//2)
                    if (bottom - top) < new_height:
                        top, bottom = max(0, (top+bottom)//2 - new_height//2), min(height,  (top+bottom)//2 + new_height//2)
                    """
                    im = im.crop((left, top, right, bottom))
                    im.save(new_file)
                elif x["name"] == "ellipse":
                    left = max(0, x["cx"] - new_width//2)
                    top = max(0, x["cy"] - new_height//2)
                    right = min(width, x["cx"] +  new_width//2)
                    bottom = min(height, x["cy"] + new_height//2)
                    im = im.crop((left, top, right, bottom))
                    im.save(new_file)
                elif x["name"] == "circle":
                    left = max(0, x["cx"] - new_width//2)
                    top = max(0, x["cy"] - new_height//2)
                    right = min(width, x["cx"] +  new_width//2)
                    bottom = min(height, x["cy"] + new_height//2)
                    im = im.crop((left, top, right, bottom))
                    im.save(new_file)

    # Creating segmented images for Benign cases
    directory = path.join(output_images_dir, "Benign")
    save_directory = path.join(output_segmented_dir, "Benign")
    
    os.makedirs(save_directory, exist_ok=True)
    print("Creating segmented images for \"Benign\" cases ........")
    for filename in os.listdir(directory):
        if cesm_only:
            if '_CM_' not in filename:
                continue
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            rows = df.loc[df['#filename'] == filename]
            shapes = rows['region_shape_attributes'].values
            i = 0
            for shape in shapes:
                x = json.loads(shape)
                i = i + 1
                new_file = os.path.join(
                    save_directory,  filename.split('.')[0] + str(i) + '.' + filename.split('.')[1])
                im = Image.open(f)
                width, height = im.size   # Get dimensions    
                if x["name"] == "polygon":
                    left = min(x["all_points_x"])
                    top = min(x["all_points_y"])
                    right = max(x["all_points_x"])
                    bottom = max(x["all_points_y"])
                    """
                    if (right - left) < new_width:
                        left, right = max(0, (right+left)//2 - new_width//2), min(width,  (right+left)//2 + new_width//2)
                    if (bottom - top) < new_height:
                        top, bottom = max(0, (top+bottom)//2 - new_height//2), min(height,  (top+bottom)//2 + new_height//2)
                    """
                    im = im.crop((left, top, right, bottom))
                    im.save(new_file)
                elif x["name"] == "ellipse":
                    left = max(0, x["cx"] - new_width//2)
                    top = max(0, x["cy"] - new_height//2)
                    right = min(width, x["cx"] +  new_width//2)
                    bottom = min(height, x["cy"] + new_height//2)
                    im = im.crop((left, top, right, bottom))
                    im.save(new_file)

                elif x["name"] == "circle":
                    left = max(0, x["cx"] - new_width//2)
                    top = max(0, x["cy"] - new_height//2)
                    right = min(width, x["cx"] +  new_width//2)
                    bottom = min(height, x["cy"] + new_height//2)
                    im = im.crop((left, top, right, bottom))
                    im.save(new_file)
    return 0


def process_vision_data(
        image_folder,
        manual_annotations_file,
        segmentation_path,
        output_dir):
    # Classify images data into Benign,Malignant,Normal
    classify_images(
        image_folder, manual_annotations_file, output_dir)

    # Segment the classified images and save the ROI tiles
    segment_images(
        segmentation_path, output_dir)


def process(
        data_path,
        image_folder=None,
        manual_annotations_file=None,
        segmentation_path=None,
        output_dir=None):
    if data_path:
        if not image_folder:
            image_folder = os.path.join(
                data_path, "CDD-CESM")
        if not manual_annotations_file:
            manual_annotations_file = os.path.join(
                data_path, "Radiology manual annotations.xlsx")
        if not segmentation_path:
            segmentation_path = os.path.join(
                data_path, "Radiology_hand_drawn_segmentations_v2.csv")

        if not output_dir:
            output_dir = data_path
            print("output-dir is not specified. Default to: {}".format(
                output_dir))

    if not image_folder:
        raise ValueError(
            "Please specify data-path or explicitly image-folder.")

    if not manual_annotations_file:
        raise ValueError(
            "Please specify data-path or explicitly manual-annotations-file.")

    if not segmentation_path:
        raise ValueError(
            "Please specify data-path or explicitly segmentation-path.")

    if not output_dir:
        raise ValueError(
            "Please specify output-dir for storing the output.")

    process_vision_data(
        image_folder,
        manual_annotations_file,
        segmentation_path,
        output_dir)


def run(args):
    process(
        data_path=args.data_path,
        image_folder=args.image_folder,
        manual_annotations_file=args.manual_annotations_file,
        segmentation_path=args.segmentation_path,
        output_dir=args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process data for Breast Cancer annotation for vision images."
    )
    parser.add_argument(
        "--data-path", "--data_path",
        type=str,
        help="Location of root of data path",
    )
    parser.add_argument(
        "--image-folder", "--image_folder",
        type=str,
        help="Location of image folder",
    )
    parser.add_argument(
        "--manual-annotations-file", "--manual_annotations_file",
        type=str,
        help="Location of manual annotations file",
    )
    parser.add_argument(
        "--segmentation-path", "--segmentation_path",
        type=str,
        help="Location of segmentation path",
    )
    parser.add_argument(
        "--output-dir", "--output_dir",
        type=str,
        help="Path to the output directory",
    )

    args = parser.parse_args()
    print(args)

    run(args)
