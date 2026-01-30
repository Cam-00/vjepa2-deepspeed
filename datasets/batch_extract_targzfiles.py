#!/usr/bin/env python3
import os
import tarfile
from pathlib import Path

def batch_extract_tar_gz(directory, output_dir=None, remove_original=False):
    """
    Batch extract all .tar.gz files in a directory.
    :param directory: Directory to search in.
    :param output_dir: Target directory for extraction (defaults to the current directory).
    :param remove_original: Whether to delete the original .tar.gz files after extraction.
    """
    directory = Path(directory).resolve()
    if output_dir is None:
        output_dir = directory
    else:
        output_dir = Path(output_dir).resolve()
        os.makedirs(output_dir, exist_ok=True)

    extracted_count = 0

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.tar.gz'):
                tar_path = Path(root) / file
                try:
                    # Extract to a subdirectory named after the file (removing .tar.gz)
                    # extract_dir = output_dir / file[:-7]
                    # os.makedirs(extract_dir, exist_ok=True)

                    # Extract to the specified path
                    extract_dir = output_dir

                    print(f"Extracting: {tar_path} -> {extract_dir}")
                    with tarfile.open(tar_path, 'r:gz') as tar:
                        for member in tar.getmembers():
                            if member.isfile():  # Process files only, skip directories
                                # Get target file path
                                dest_path = os.path.join(extract_dir, os.path.basename(member.name))

                                # Skip if file already exists
                                if os.path.exists(dest_path):
                                    # print(f"Skipping existing file: {dest_path}")
                                    continue

                                # Get filename (strip path)
                                filename = os.path.basename(member.name)
                                # Modify member name to extract directly into output_dir
                                member.name = filename
                                tar.extract(member, path=extract_dir)
                        # tar.extractall(path=extract_dir)

                    extracted_count += 1

                    if remove_original:
                        os.remove(tar_path)
                        print(f"Original file deleted: {tar_path}")

                except Exception as e:
                    print(f"Extraction failed: {tar_path} - {str(e)}")

    print(f"\nExtraction complete! Processed {extracted_count} files.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Batch extract .tar.gz files')
    parser.add_argument('directory', help='Directory to search for files')
    parser.add_argument('-o', '--output', help='Target directory for extraction (default is the current directory)')
    parser.add_argument('-r', '--remove', action='store_true', help='Delete original files after extraction')
    args = parser.parse_args()

    # args.directory = '/mnt/mydisk/Cam/Kinetics_700-2020/Kinetics_700-2020/kinetics-dataset/k700-2020_targz/train'
    # args.output = '/mnt/mydisk/Cam/Kinetics_700-2020/Kinetics_700-2020/kinetics-dataset/k700-2020/train'

    batch_extract_tar_gz(args.directory, args.output, args.remove)
