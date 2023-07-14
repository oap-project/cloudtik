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

import argparse
import os
import tempfile
import zipfile

import docx2txt

from cloudtik.runtime.ai.util.utils import remove_dir
from disease_prediction.utils import _get_data_api


def read_right_and_left(tx):
    tx_right, tx_left = "", ""
    if "Right Breast:" in tx and "Left Breast:" in tx:
        tx = tx.split("Left Breast:")
        tx_right = [
            i
            for i in tx[0].split("Right Breast:")[1].splitlines()
            if ("ACR C:" not in i and i != "")
        ]
        tx_left = [i for i in tx[1].splitlines() if ("ACR C:" not in i and i != "")]

    elif "Right Breast:" in tx and "Left Breast:" not in tx:
        tx = tx.split("Right Breast:")[1].splitlines()
        tx_right = [i for i in tx if i != ""]

    elif "Right Breast:" not in tx and "Left Breast:" in tx:
        tx = tx.split("Left Breast:")[1].splitlines()
        tx_left = [i for i in tx if i != ""]

    return tx_right, tx_left


def read_content(file_content):
    annotation = file_content.split("OPINION:")  
    mm_revealed = annotation[0].split("REVEALED:")[1]
    mm_revealed_right, mm_revealed_left = read_right_and_left(mm_revealed)

    optinion = annotation[1].split("CONTRAST ENHANCED SPECTRAL MAMMOGRAPHY REVEALED:")
    ces_mm_revealed = optinion[1]
    optinion = optinion[0]
    optinion_right, optinion_left = read_right_and_left(optinion)

    ces_mm_revealed_right, ces_mm_revealed_left = read_right_and_left(ces_mm_revealed)

    return (
        mm_revealed_right,
        mm_revealed_left,
        optinion_right,
        optinion_left,
        ces_mm_revealed_right,
        ces_mm_revealed_left,
    )


def add_df_log(df, dict_text, manual_annotations, f_id):
    for side in manual_annotations.Side.unique():
        for mm_type in manual_annotations.Type.unique().tolist() + ["OP"]:
            text_list = dict_text[mm_type + "_" + side]
            df_temp = manual_annotations[
                (manual_annotations.Patient_ID == int(f_id))
                & (manual_annotations.Side == side)
                & (manual_annotations.Type == mm_type)
            ]
            image_name = df_temp.Image_name.tolist()

            if mm_type == "OP":
                label = [None]
            else:
                label = df_temp["Pathology Classification/ Follow up"].unique().tolist()

            if len(label) == 1:
                df.loc[len(df)] = [
                    f_id,
                    image_name,
                    side,
                    mm_type,
                    label[0],
                    " ".join(text_list),
                ]

    return df


def label_correction(df, pd):
    label_column = "label"
    data_column = "symptoms"
    patient_id = "Patient_ID"

    df_new = pd.DataFrame(columns=[label_column, data_column, patient_id])
    for i in df[patient_id].unique():
        annotation = " ".join(df[df[patient_id].isin([i])][data_column].to_list())
        temp_labels = [
            label_indx
            for label_indx in df[df[patient_id] == i][label_column].unique()
            if label_indx is not None
        ]

        if len(temp_labels) == 1:
            df_new.loc[len(df_new)] = [temp_labels[0], annotation, i]
        elif len(temp_labels) > 1:
            # CM images are substracted images, if available use the labels of the CM not DM
            # {patient number}_{breast side}_{image type}_{image view}; example ‘P1_L_CM_MLO’
            # (DM)   Digital mammography
            # (CESM) Contrast-enhanced spectral mammography

            df_temp = df[df[patient_id].isin([i])]

            if "CESM" in df_temp.Type.to_list():
                new_label = df_temp[df_temp.Type == "CESM"].label.to_list()[0]
                df_new.loc[len(df_new)] = [new_label, annotation, i]

        else:
            pass

    return df_new


def unzip_file(medical_reports_zip_file, output_dir):
    medical_reports_folder_name = os.path.basename(
        medical_reports_zip_file).split(".zip")[0].strip()
    medical_reports_folder = os.path.join(
        output_dir, medical_reports_folder_name)
    remove_dir(medical_reports_folder)

    with zipfile.ZipFile(medical_reports_zip_file, "r") as zip_ref:
        zip_ref.extractall(output_dir)  # ( medical_reports_folder )

    return medical_reports_folder


def save_annotation_file(df, output_file):
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_file, index=False)
    print("----- file is saved here :", output_file)


def _process(
    medical_reports_zip_file,
    manual_annotations_file,
    output_annotations_file,
    temp_dir,
    data_api,
):
    print("----- Starting the data preprocessing -----")
    pd = data_api.pandas()
    manual_annotations = pd.read_excel(manual_annotations_file, sheet_name="all")

    medical_reports_folder = unzip_file(
        medical_reports_zip_file, temp_dir)

    df = pd.DataFrame(columns=["ID", "Image", "Side", "Type", "label", "symptoms"])
    
    for f in os.listdir(medical_reports_folder):
        DM_R, DM_L, OP_R, OP_L, CESM_R, CESM_L = "", "", "", "", "", ""
        f_id = f.split(".docx")[0].split("P")[1]

        try:
            file_content = docx2txt.process(os.path.join(medical_reports_folder, f))
        except Exception as e:
            # TODO: shall raise exception
            Warning(e)

        DM_R, DM_L, OP_R, OP_L, CESM_R, CESM_L = read_content(file_content)
        dict_text = {
            "DM_R": DM_R,
            "DM_L": DM_L,
            "OP_R": OP_R,
            "OP_L": OP_L,
            "CESM_R": CESM_R,
            "CESM_L": CESM_L,
        }

        df = add_df_log(df, dict_text, manual_annotations, f_id)

    df["Patient_ID"] = [
        "".join([str(df.loc[i, "ID"]), df.loc[i, "Side"]]) for i in df.index
    ]
    
    df = label_correction(df, pd)
    save_annotation_file(df, output_annotations_file)


def process(
        data_path,
        data_api,
        medical_reports_folder=None,
        manual_annotations_file=None,
        output_annotations_file=None,
        temp_dir=None):
    if data_path:
        if not medical_reports_folder:
            medical_reports_folder = os.path.join(
                data_path, "Medical reports for cases .zip")
        if not manual_annotations_file:
            manual_annotations_file = os.path.join(
                data_path, "Radiology manual annotations.xlsx")
        if not output_annotations_file:
            output_annotations_file = os.path.join(
                data_path, "annotation", "annotation.csv")

    if not medical_reports_folder:
        raise ValueError(
            "Please specify data-path or explicitly medical-reports-folder.")

    if not manual_annotations_file:
        raise ValueError(
            "Please specify data-path or explicitly manual-annotations-file.")

    if not output_annotations_file:
        raise ValueError(
            "Please specify data-path or explicitly output-annotations-file.")

    if not temp_dir:
        # for single node, get get a default temp dir from /tmp
        temp_dir = tempfile.mkdtemp()
        print("temp-dir is not specified. Default to: {}".format(
            temp_dir))
    _process(
        medical_reports_folder,
        manual_annotations_file,
        output_annotations_file,
        temp_dir=temp_dir,
        data_api=data_api,
    )


def run(args):
    data_api = _get_data_api(args)
    process(
        data_path=args.data_path,
        data_api=data_api,
        medical_reports_folder=args.medical_reports_folder,
        manual_annotations_file=args.manual_annotations_file,
        output_annotations_file=args.output_annotations_file,
        temp_dir=args.temp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process document data for Breast Cancer annotation."
    )
    parser.add_argument(
        "--data-path", "--data_path",
        type=str,
        help="Location of root of data path",
    )
    parser.add_argument(
        "--temp-dir", "--temp_dir",
        type=str,
        help="The path to the intermediate data")
    parser.add_argument(
        "--medical-reports-folder", "--medical_reports_folder",
        type=str,
        help="Location of medical reports for cases",
    )  
    parser.add_argument(
        "--manual-annotations-file", "--manual_annotations_file",
        type=str,
        help="Location of manual annotations file",
    )
    parser.add_argument(
        "--output-annotations-file", "--output_annotations_file",
        type=str,
        help="Name of the output annotation file",
    )

    args = parser.parse_args()
    print(args)

    run(args)
