import os
import pandas as pd
import torch


def mimic_cxr_image_path(image_dir, subject_id, study_id, dicom_id, ext='dcm'):
    return os.path.join(image_dir, 'p' + str(subject_id)[:2], 'p' + str(subject_id),
                        's' + str(study_id), str(dicom_id) + '.' + ext)


def mimic_cxr_text_path(image_dir, subject_id, study_id, ext='txt'):
    return os.path.join(image_dir, 'p' + str(subject_id)[:2], 'p' + str(subject_id),
                        's' + str(study_id) + '.' + ext)

def enumerated_save_path(save_dir, save_name, extension):
    save_path = os.path.join(save_dir, save_name + extension)
    assert '.' in extension, 'No period in extension.'
    if os.path.isfile(save_path):
        count = 2
        while True:
            save_path = os.path.join(save_dir, save_name + "_" + str(count) + extension)
            count += 1
            if not os.path.isfile(save_path):
                break

    return save_path