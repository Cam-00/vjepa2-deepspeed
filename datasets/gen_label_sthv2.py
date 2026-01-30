# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# ------------------------------------------------------
# Code adapted from https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py
# processing the raw data of the video Something-Something-V2

# 将Something-Something-V2原始webm格式视频, 做一个类别标签映射, 生成

import os
import json

if __name__ == '__main__':
    # dataset_name = 'D:\\Datasets\\something_v2_small\\annotations\\something-something-v2'  # For Windows
    # output_folder = 'D:\\Datasets\\something_v2_small'  # For Windows

    dataset_name = '/mnt/windows/Datasets/Something_Something_v2'  # For Linux
    output_folder = '/home/cam/LLM/vjepa2/datasets/Something_Something_v2'  # For Linux

    # Load category label file (Format: {"action_name": category_id})
    with open('%s/annotations/something-something-v2-labels.json' % dataset_name) as f:
        data = json.load(f)

    # Extract ordered list of categories
    categories = []
    for i, (cat, idx) in enumerate(data.items()):
        assert i == int(idx)  # Ensure the rank/index is correct
        categories.append(cat)

    # Write category list to category.csv (one category per line)
    with open('%s/category.csv' % output_folder, 'w') as f:
        f.write('\n'.join(categories))

    # Create mapping dictionary from category name to index
    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    # Define input and output file lists
    files_input = [
        '%s/annotations/something-something-v2-validation.json' % dataset_name,
        '%s/annotations/something-something-v2-train.json' % dataset_name,
        '%s/annotations/something-something-v2-test.json' % dataset_name
    ]
    files_output = [
        '%s/ssv2_val_paths.csv' % output_folder,
        '%s/ssv2_train_paths.csv' % output_folder,
        '%s/ssv2_test_paths.csv' % output_folder
    ]

    for (filename_input, filename_output) in zip(files_input, files_output):
        # Encoding must be specified (utf-8) to avoid potential GBK decoding errors
        with open(filename_input, encoding="utf-8") as f:
            data = json.load(f)

        folders = []  # Stores video ID strings
        idx_categories = []  # Stores category labels corresponding to video IDs

        for item in data:
            folders.append(item['id'])
            if 'test' not in filename_input:
                # Remove brackets from template and map to category index
                idx_categories.append(dict_categories[item['template'].replace('[', '').replace(']', '')])
            else:
                # Default label for test set
                idx_categories.append(0)

        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]

            # counting the number of frames in each video folders
            # dir_files = os.listdir(os.path.join('%s\\20bn-something-something-v2' % output_folder, curFolder))
            # output.append('%s %d %d' % (curFolder, len(dir_files), curIDX))

            # Map action category labels for original ssv2 webm videos
            # This method is also applicable for mapping labels to subsets of original webm videos
            curFolder += '.webm'
            video_path = os.path.join('%s/20bn-something-something-v2' % dataset_name, curFolder)

            if os.path.exists(video_path):
                output.append('%s %d' % (video_path, curIDX))
                print('%d/%d' % (i, len(folders)))

        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))
