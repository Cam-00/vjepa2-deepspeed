import csv


def simple_process_txt_to_csv(txt_file, csv_file, start=''):
    """
    Simplified version: Process a TXT file and convert it to CSV.

    Args:
        txt_file: Path to the input TXT file.
        csv_file: Path to the output CSV file.
        start: Prefix text to add at the beginning of each line.
        end: Suffix text to add at the end of each line.
    """

    # Read and process TXT
    with open(txt_file, 'r', encoding='utf-8') as f, open(csv_file, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.writer(csvf)
        # Read all lines and strip trailing whitespace
        lines = [line.rstrip() for line in f]

        # Process each line: append the content before the first '/' to the end
        for line in lines:
            if '/' in line:
                # Extract the content before the first '/'
                label = line.split('/')[0]
                # Append to the end of the line
                processed_line = start + line + ' ' + label
            else:
                # If no '/' is found, copy as is
                processed_line = start + line
            writer.writerow([processed_line])

    print(f"Done! Converted {txt_file} to {csv_file}")
    print(f"Processed {len(lines)} lines, added prefix: '{start}'")


# Usage Example
if __name__ == "__main__":
    # Modify these parameters as needed
    input_txt = "/home/cam/dataset/ImageNet_1K_train.txt"  # Your TXT file
    output_csv = "/home/cam/LLM/vjepa2_deepspeed/datasets/imagenet_1k/imagenet_1k_train_path.csv"  # Output CSV file
    prefix = "/cloud/cloud-ssd1/dataset/ImageNet-1K/ImageNet-1K/train/"  # Text to add at the beginning of each line
    # suffix = " "  # Text to add at the end of each line

    # Execute processing
    simple_process_txt_to_csv(input_txt, output_csv, prefix)