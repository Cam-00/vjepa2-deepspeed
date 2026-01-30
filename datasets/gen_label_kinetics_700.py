import csv


def simple_process_txt_to_csv(txt_file, csv_file, start=''):
    """
    Simplified version: Process a TXT file and convert it to CSV.

    Args:
        txt_file: Input TXT file path.
        csv_file: Output CSV file path.
        start: Prefix text to be added at the beginning of each line.
    """

    # Read and process TXT
    with open(txt_file, 'r', encoding='utf-8') as f, open(csv_file, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.writer(csvf)
        # Read all lines and process
        lines = [line.rstrip() for line in f]  # Remove trailing \n from each line

        # Process each line: add prefix to the beginning
        for line in lines:
            processed_line = start + line
            writer.writerow([processed_line])

    print(f"Done! Converted {txt_file} to {csv_file}")
    print(f"Processed {len(lines)} lines, added prefix: '{start}'")


# Usage Example
if __name__ == "__main__":
    # Modify these parameters
    input_txt = "/home/cam/dataset/kinetics700_val_list_videos.txt"  # Your TXT file
    output_csv = "/home/cam/LLM/vjepa2_deepspeed/datasets/kinetics_700/kinetics_700_val_path.csv"  # Output CSV file
    prefix = "/cloud/cloud-ssd1/dataset/Kinetics_700/videos/"  # Prefix text added to each line
    # suffix = " "  # Suffix text added to each line

    # Execute processing
    simple_process_txt_to_csv(input_txt, output_csv, prefix)