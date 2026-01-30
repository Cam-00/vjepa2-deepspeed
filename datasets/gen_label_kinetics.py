# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# ------------------------------------------------------
# Code adapted from https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py
import os
dataset_path = '/mnt/mydisk/Cam/Kinetics_700-2020/Kinetics_700-2020/kinetics-dataset'
label_path = '/home/cam/LLM/vjepa2/datasets/kinetics_700-2020'

if __name__ == '__main__':
    with open('%s/labels.csv' % label_path) as f:
        categories = f.readlines()
        categories = [c.strip().replace(' ', '_').replace('"', '').replace('(', '').replace(')', '').replace("'", '') for c in categories]
    assert len(set(categories)) == 700
    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    # print(dict_categories)

    # Specify input and output file path lists
    output_folder = label_path
    # files_input = ['%s/k700-2020/annotations/val.csv' % dataset_path, '%s/k700-2020/annotations/train.csv' % dataset_path, '%s/k700-2020/annotations/test.csv' % dataset_path]
    # files_output = ['%s/k700_val_paths.csv' % output_folder, '%s/k700_train_paths.csv' % output_folder, '%s/k700_test_paths.csv' % output_folder]
    files_input = ['%s/k700-2020/annotations/test.csv' % dataset_path]
    files_output = ['%s/k700_test_paths.csv' % output_folder]

    for (filename_input, filename_output) in zip(files_input, files_output):
        count_cat = {k: 0 for k in dict_categories.keys()}
        with open(os.path.join(label_path, filename_input)) as f:
            lines = f.readlines()[1:]
        folders = []
        idx_categories = []
        categories_list = []
        for line in lines:
            line = line.rstrip()
            items = line.split(',')
            if 'test' not in filename_input:
                folders.append(items[1] + '_' + items[2].zfill(6) + '_' + items[3].zfill(6))
                this_catergory = items[0].replace(' ', '_').replace('"', '').replace('(', '').replace(')', '').replace(
                    "'", '')
                idx_categories.append(dict_categories[this_catergory])
                count_cat[this_catergory] += 1
            else:
                folders.append(items[0] + '_' + items[1].zfill(6) + '_' + items[2].zfill(6))
                this_catergory = 0
                idx_categories.append(0)

            categories_list.append(this_catergory)

        # print(max(count_cat.values()))

        assert len(idx_categories) == len(folders)
        # missing_folders = []
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]

            # counting the number of frames in each video folders
            # img_dir = os.path.join(dataset_path, categories_list[i], curFolder)
            # if not os.path.exists(img_dir):
            #     missing_folders.append(img_dir)
            #     # print(missing_folders)
            # else:
            #     dir_files = os.listdir(img_dir)
            #     output.append('%s %d %d'%(os.path.join(categories_list[i], curFolder), len(dir_files), curIDX))
            # print('%d/%d, missing %d'%(i, len(folders), len(missing_folders)))

            curFolder += '.mp4'
            video_path = os.path.join('%s/k700-2020/test' % dataset_path, curFolder)
            if os.path.exists(video_path):
                output.append('%s %d' % (video_path, curIDX))
                # print('%d/%d' % (i, len(folders)))

        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))
        # with open(os.path.join(label_path, 'missing_' + filename_output),'w') as f:
        #     f.write('\n'.join(missing_folders))
