import os
import numpy as np
import pdb
import pandas as pd
from PIL import Image
import cv2
from patchify import patchify
from tqdm import tqdm

def get_dataset_dict(classes):
    column_dict = {'image_path': ''}
    class_dict = {key: '' for key in classes}
    column_dict.update(class_dict)
    return column_dict

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
        
        image_dir = os.path.join(root_directory,f"{phase}-org-img")
        mask_dir = os.path.join(root_directory,f"{phase}-label-img")
    else:
        if phase == 'train':
            root_directory = config['original_data']['train_directory']
        else:
            root_directory = config['original_data']['val_directory']

        image_dir = os.path.join(root_directory,'images')
        mask_dir = os.path.join(root_directory,config['original_data']['label_folder_prefix'])
        
    for image_file in tqdm(os.listdir(image_dir),desc='Processing Images',leave=False):
        mask_file = f"{image_file.split('.')[0]}_{config['original_data']['label_file_suffix']}.{config['original_data']['label_ext']}"
        ext = image_file.split('.')[-1]
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path, 1)
        mask_path = os.path.join(mask_dir,mask_file)
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
                single_patch_mask = cv2.cvtColor(single_patch_mask,cv2.COLOR_RGB2GRAY)
                for cls_id, label in enumerate(classes):
                    if (single_patch_mask == cls_id).any():
                        blank_image = np.zeros_like(single_patch_img)
                        coords = np.argwhere(single_patch_mask == cls_id)
                        for x, y in coords:
                            blank_image[x, y] = single_patch_img[x, y]
                        patch_filename = f"{base_name}_patch_{i}_{j}_{label}.{ext}"
                        patch_path = os.path.join(output_images_directory, patch_filename)
                        cv2.imwrite(patch_path, blank_image)
                        dataset_dict['image_path']=patch_path
                        dataset_dict[classes[cls_id]]=1
                    else:
                        dataset_dict[classes[cls_id]]=0
                rows.append(dataset_dict)
    dataset_df = pd.DataFrame(rows)
    dataset_df.to_csv(csv_file_path, index=False)
    print(f"CSV file saved to {csv_file_path}")

# if __name__ == "__main__":
#     config = {
#         'original_data': {
#             'train_directory': 'path/to/train_directory',
#             'val_directory': 'path/to/val_directory',
#             'label_folder_prefix': 'labels'
#         }
#     }
#     output_directory = 'path/to/output_directory'
#     classes = [0, 1, 2]  # Replace with actual class indices present in your masks
#     patch_size = 256  # Adjust the patch size as needed

#     process_and_patch_images(config, output_directory, classes, patch_size)
