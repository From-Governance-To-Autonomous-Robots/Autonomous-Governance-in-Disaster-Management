import os
import argparse
from utils.utils import load_config

def main(args):
    config = load_config(args.config)
    
    # create multilabel_dataset directories 
    multi_label_dir = os.path.join(config['original_data']['root_dir'],'multilabel_dataset')
    os.makedirs(multi_label_dir,exist_ok=True)
    train_multi_dir = os.path.join(multi_label_dir,'train')
    os.makedirs(train_multi_dir,exist_ok=True)
    val_multi_dir = os.path.join(multi_label_dir,'val')
    os.makedirs(val_multi_dir,exist_ok=True)
    
    # process and create the datasets - patch and store csv 
    if (config['paths']['task']).startswith('drone_'):
        from data.drone_patching_logic import process_and_patch_images 
    else:
        from data.patching_logic import process_and_patch_images
    
    process_and_patch_images(
        config=config,
        output_directory=train_multi_dir,
        classes=config['original_data']['gt_labels'],
        patch_size=config['original_data']['patch_size'],
        phase='train'
    )
    
    process_and_patch_images(
        config=config,
        output_directory=val_multi_dir,
        classes=config['original_data']['gt_labels'],
        patch_size=config['original_data']['patch_size'],
        phase='val'
    )
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="prepare dataset for multilabel classification")
    parser.add_argument('-config',type=str,default="/home/julian/git-repo/juliangdz/GovernanceIRP/Autonomous-Governance-in-Disaster-Management/post_disaster_classification/configs/drone_config.yaml")
    args = parser.parse_args()
    
    main(args)