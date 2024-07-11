import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from patchify import patchify
import pandas as pd

def get_dataset_dict(classes):
    dataset_dict = {cls: 0 for cls in classes}
    dataset_dict['image_path'] = ''
    return dataset_dict

def process_and_patch_images(config, output_directory: str, classes: list, patch_size: int = 256, phase: str = 'train'):
    output_images_directory = os.path.join(output_directory, 'images')
    os.makedirs(output_images_directory, exist_ok=True)
    output_mask_folder = os.path.join(output_directory, 'masks')
    os.makedirs(output_mask_folder, exist_ok=True)
    log_folder = os.path.join(output_directory, 'logs')
    os.makedirs(log_folder, exist_ok=True)

    csv_file_path = os.path.join(log_folder, 'dataset.csv')
    rows = []

    if config['paths']['task'].startswith('drone_'):
        if phase == 'train':
            root_directory = config['original_data']['train_directory']
        else:
            root_directory = config['original_data']['val_directory']
        
        image_dir = os.path.join(root_directory, f"{phase}-org-img")
        mask_dir = os.path.join(root_directory, f"{phase}-label-img")
    else:
        if phase == 'train':
            root_directory = config['original_data']['train_directory']
        else:
            root_directory = config['original_data']['val_directory']

        image_dir = os.path.join(root_directory, 'images')
        mask_dir = os.path.join(root_directory, config['original_data']['label_folder_prefix'])

    for image_file in tqdm(os.listdir(image_dir), desc='Processing Images', leave=False):
        mask_file = f"{image_file.split('.')[0]}_{config['original_data']['label_file_suffix']}.{config['original_data']['label_ext']}"
        ext = image_file.split('.')[-1]
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path, 1)
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, 0)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        SIZE_X = (image.shape[1] // patch_size) * patch_size
        SIZE_Y = (image.shape[0] // patch_size) * patch_size
        image = Image.fromarray(image)
        image = image.crop((0, 0, SIZE_X, SIZE_Y))
        image = np.array(image)

        patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)
        patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)
        base_name = os.path.splitext(image_file)[0]

        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                dataset_dict = get_dataset_dict(classes)
                single_patch_img = patches_img[i, j, 0]
                single_patch_mask = patches_mask[i, j, 0]
                single_patch_mask = cv2.cvtColor(single_patch_mask, cv2.COLOR_RGB2GRAY)
                unique_values = np.unique(single_patch_mask)

                patch_filename = f"{base_name}_patch_{i}_{j}.{ext}"
                patch_path = os.path.join(output_images_directory, patch_filename)
                cv2.imwrite(patch_path, single_patch_img)
                dataset_dict['image_path'] = patch_path

                for cls_id, label in enumerate(classes):
                    if cls_id in unique_values:
                        dataset_dict[label] = 1
                    else:
                        dataset_dict[label] = 0

                rows.append(dataset_dict)

    dataset_df = pd.DataFrame(rows)
    dataset_df.to_csv(csv_file_path, index=False)
    print(f"CSV file saved to {csv_file_path}")
