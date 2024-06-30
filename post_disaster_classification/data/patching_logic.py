import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from patchify import patchify

def get_dataset_dict(classes):
    column_dict = {'image_path':[]}
    col = column_dict.copy()
    class_dict = {key: [] for key in classes}
    return col.update(class_dict)

def process_and_patch_images(config, output_directory:str, classes:list, patch_size:int=256,phase:str='train'):
    output_images_directory = os.path.join(output_directory, 'images')
    os.makedirs(output_images_directory, exist_ok=True)
    output_mask_folder = os.path.join(output_directory, 'masks')
    os.makedirs(output_mask_folder, exist_ok=True)
    log_folder = os.path.join(output_directory, 'logs')
    os.makedirs(log_folder, exist_ok=True)
    
    csv_file_path = os.path.join(log_folder, 'dataset.csv')
    dataset_dict = get_dataset_dict(classes)
    
    if phase == 'train':
        root_directory = config['original_data']['train_directory']
    else:
        root_directory = config['original_data']['val_directory']
    
    for path, subdirs, files in os.walk(root_directory):
        dirname = path.split(os.path.sep)[-1]
        
        if dirname == 'images':
            for image_name in files:
                ext = image_name.split('.')[-1]
                image_path = os.path.join(path, image_name)
                image = cv2.imread(image_path, 1)
                
                SIZE_X = (image.shape[1] // patch_size) * patch_size
                SIZE_Y = (image.shape[0] // patch_size) * patch_size
                image = Image.fromarray(image)
                image = image.crop((0, 0, SIZE_X, SIZE_Y))
                image = np.array(image)
                
                print(f"Now patchifying image: {image_path}")
                patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)
                
                base_name = os.path.splitext(image_name)[0]
                
                for i in range(patches_img.shape[0]):
                    for j in range(patches_img.shape[1]):
                        single_patch_img = patches_img[i, j, 0]
                        patch_filename = f"{base_name}_patch_{i}_{j}.{ext}"
                        patch_path = os.path.join(output_images_directory, patch_filename)
                        cv2.imwrite(patch_path, single_patch_img)
                        dataset_dict['image_path'].append(patch_path)
        
        elif dirname == config['original_data']['label_folder_prefix']:
            for mask_name in files:
                mask_path = os.path.join(path, mask_name)
                ext = mask_name.split('.')[-1]
                mask = cv2.imread(mask_path, 1)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                
                SIZE_X = (mask.shape[1] // patch_size) * patch_size
                SIZE_Y = (mask.shape[0] // patch_size) * patch_size
                mask = Image.fromarray(mask)
                mask = mask.crop((0, 0, SIZE_X, SIZE_Y))
                mask = np.array(mask)
                
                print(f"Now patchifying mask: {mask_path}")
                patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)
                
                base_name = os.path.splitext(mask_name)[0]
                
                for i in range(patches_mask.shape[0]):
                    for j in range(patches_mask.shape[1]):
                        single_patch_mask = patches_mask[i, j, 0]
                        patch_filename = f"{base_name}_patch_{i}_{j}.{ext}"
                        patch_path = os.path.join(output_mask_folder, patch_filename)
                        cv2.imwrite(patch_path, single_patch_mask)
                        
                        unique_values = np.unique(single_patch_mask)
                        for cls_id in range(len(classes)):
                            if cls_id in unique_values:
                                dataset_dict[classes[cls_id]].append(1)
                            else:
                                dataset_dict[classes[cls_id]].append(0)
    
    datset_df = pd.DataFrame(dataset_dict)
    datset_df.to_csv(csv_file_path, index=False)
    print(f"CSV file saved to {csv_file_path}")

# if __name__ == "__main__":
#     root_directory = 'path/to/root_directory'
#     output_image_folder = 'path/to/output_image_folder'
#     output_mask_folder = 'path/to/output_mask_folder'
#     log_folder = 'path/to/log_folder'
    
#     classes = [0, 1, 2]  # Replace with actual class indices present in your masks
#     patch_size = 256  # Adjust the patch size as needed
    
#     process_and_patch_images(root_directory, output_image_folder, output_mask_folder, log_folder, classes, patch_size)
