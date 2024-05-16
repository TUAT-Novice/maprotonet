import copy
import torch
import torchio as tio
import numpy as np

from pathlib import Path
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Tuple


def load_data(data_path):
    name_mapping = np.genfromtxt(f'{data_path}/name_mapping.csv', delimiter=',', dtype=None, encoding='utf8')
    subj_ids, labels = name_mapping[1:, -1], name_mapping[1:, 0]
    labels = label_binarize(np.array(labels == 'HGG', dtype=int), classes=[0, 1, np.inf])[:, :-1]

    subjs = []
    for subj_id, label in zip(subj_ids, labels):
        try:
            # BraTS2020
            subj = tio.Subject(t1=tio.ScalarImage(f'{data_path}/{subj_id}/{subj_id}_t1.nii.gz'),
                               t1ce=tio.ScalarImage(f'{data_path}/{subj_id}/{subj_id}_t1ce.nii.gz'),
                               t2=tio.ScalarImage(f'{data_path}/{subj_id}/{subj_id}_t2.nii.gz'),
                               flair=tio.ScalarImage(f'{data_path}/{subj_id}/{subj_id}_flair.nii.gz'),
                               seg=tio.LabelMap(f'{data_path}/{subj_id}/{subj_id}_seg.nii.gz'),
                               subj_id=subj_id, label=label)
        except:
            # BraTS2019, BraTS2018
            label_dict = {1: 'HGG', 0: 'LGG'}
            subj = tio.Subject(t1=tio.ScalarImage(f'{data_path}/{label_dict[label[1]]}/{subj_id}/{subj_id}_t1.nii.gz'),
                               t1ce=tio.ScalarImage(f'{data_path}/{label_dict[label[1]]}/{subj_id}/{subj_id}_t1ce.nii.gz'),
                               t2=tio.ScalarImage(f'{data_path}/{label_dict[label[1]]}/{subj_id}/{subj_id}_t2.nii.gz'),
                               flair=tio.ScalarImage(f'{data_path}/{label_dict[label[1]]}/{subj_id}/{subj_id}_flair.nii.gz'),
                               seg=tio.LabelMap(f'{data_path}/{label_dict[label[1]]}/{subj_id}/{subj_id}_seg.nii.gz'),
                               subj_id=subj_id, label=label)
        subjs.append(subj)
    return np.array(subjs), labels


def get_batch_images_and_size(batch: Dict) -> Tuple[int, List[str], List[str]]:
    """Get batch size and lists of image names and other names in a batch.

    Args:
        batch: Dictionary generated by a :class:`torch.utils.data.DataLoader`
        extracting data from a :class:`torchio.SubjectsDataset`.

    Raises:
        RuntimeError: If the batch does not seem to contain any dictionaries
        that seem to represent a :class:`torchio.Image`.
    """
    batch_size, image_names, other_names = 0, [], []
    for key, value in batch.items():
        if isinstance(value, dict) and tio.constants.DATA in value:
            if batch_size > 0:
                assert batch_size == len(value[tio.constants.DATA]), 'The batch size is not unique'
            else:
                batch_size = len(value[tio.constants.DATA])
            image_names.append(key)
        else:
            if batch_size > 0:
                assert batch_size == len(value), 'The batch size is not unique'
            else:
                batch_size = len(value)
            other_names.append(key)
    if not image_names:
        raise RuntimeError('The batch does not seem to contain any images')
    return batch_size, image_names, other_names


def get_subjects_from_batch(batch: Dict) -> List:
    """Get list of subjects from collated batch.

    Args:
        batch: Dictionary generated by a :class:`torch.utils.data.DataLoader`
        extracting data from a :class:`torchio.SubjectsDataset`.
    """
    subjects = []
    batch_size, image_names, other_names = get_batch_images_and_size(batch)
    for i in range(batch_size):
        subject_dict = {}
        for image_name in image_names:
            image_dict = batch[image_name]
            data = image_dict[tio.constants.DATA][i]
            affine = image_dict[tio.constants.AFFINE][i]
            path = Path(image_dict[tio.constants.PATH][i])
            is_label = image_dict[tio.constants.TYPE][i] == tio.constants.LABEL
            klass = tio.LabelMap if is_label else tio.ScalarImage
            image = klass(tensor=data, affine=affine, filename=path.name)
            subject_dict[image_name] = image
        for other_name in other_names:
            subject_dict[other_name] = batch[other_name][i]
        subject = tio.Subject(subject_dict)
        if tio.constants.HISTORY in batch:
            applied_transforms = batch[tio.constants.HISTORY][i]
            for transform in applied_transforms:
                transform.add_transform_to_subject_history(subject)
        subjects.append(copy.deepcopy(subject))
    return subjects


def preprocess(data_loader):
    x = []
    percent = 10
    print("Preprocessing Dataset:")
    for b, subjs_batch in enumerate(data_loader):
        x += get_subjects_from_batch(subjs_batch)
        while (b + 1) / len(data_loader) >= percent / 100:
            print(f"---{percent}%", end='', flush=True)
            percent += 10
    print(" Finished.")
    return np.array(x)


def load_subjs_batch(subjs_batch):
    if isinstance(subjs_batch, list):
        return subjs_batch
    else:
        data = torch.cat((subjs_batch['t1']['data'], subjs_batch['t1ce']['data'], subjs_batch['t2']['data'], subjs_batch['flair']['data']), dim=1)
        return data, subjs_batch['label'], subjs_batch['seg']['data']