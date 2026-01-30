import csv
import os


def replace_dataset_path(input_csv, output_csv=None):
    """
    Replace path prefix '/mnt/windows/Datasets' with '/cloud/cloud-ssd1/dataset' in a CSV file.

    Args:
        input_csv: Path to the input CSV file.
        output_csv: Path to the output CSV file. If None, it is generated automatically.
    """

    if output_csv is None:
        # Automatically generate output filename
        base_name = os.path.splitext(input_csv)[0]
        output_csv = f"{base_name}_replaced.csv"

    # Define paths for replacement
    target_path = "/mnt/windows/Datasets"
    replacement_path = "/cloud/cloud-ssd1/dataset"

    total_rows = 0
    replaced_count = 0

    try:
        with open(input_csv, 'r', encoding='utf-8') as f_in, \
                open(output_csv, 'w', newline='', encoding='utf-8') as f_out:

            # Create CSV reader and writer
            reader = csv.reader(f_in)
            writer = csv.writer(f_out)

            for row in reader:
                new_row = []
                for cell in row:
                    # Replace the target path
                    if target_path in cell:
                        new_cell = cell.replace(target_path, replacement_path)
                        replaced_count += 1
                    else:
                        new_cell = cell
                    new_row.append(new_cell)

                writer.writerow(new_row)
                total_rows += 1

        print("=" * 60)
        print("Path replacement completed!")
        print("=" * 60)
        print(f"Input file: {input_csv}")
        print(f"Output file: {output_csv}")
        print(f"Total rows processed: {total_rows}")
        print(f"Number of rows modified: {replaced_count}")
        print(f"Rule: '{target_path}' → '{replacement_path}'")

        # Show processing examples
        print("\nProcessing Examples:")
        # Re-open files to display the first 3 rows
        with open(input_csv, 'r', encoding='utf-8') as f:
            original_lines = []
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i < 3:
                    original_lines.append(row)
                else:
                    break

        with open(output_csv, 'r', encoding='utf-8') as f:
            replaced_lines = []
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i < 3:
                    replaced_lines.append(row)
                else:
                    break

        for i in range(min(len(original_lines), len(replaced_lines))):
            print(f"\nRow {i + 1}:")
            print(f"  Original: {original_lines[i]}")
            print(f"  Modified: {replaced_lines[i]}")

        return True

    except FileNotFoundError:
        print(f"Error: File not found {input_csv}")
        return False
    except Exception as e:
        print(f"Processing error: {e}")
        return False


def process_multiple_files(file_list):
    """Batch process multiple CSV files."""
    for csv_file in file_list:
        if os.path.exists(csv_file):
            print(f"\nProcessing: {csv_file}")
            replace_dataset_path(csv_file)
        else:
            print(f"Warning: File does not exist {csv_file}")


def simple_one_liner():
    """Minimal version for single file processing."""
    input_file = "input.csv"  # Modify to your filename
    output_file = "output.csv"

    target = "/mnt/windows/Datasets"
    replacement = "/cloud/cloud-ssd1/dataset"

    with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        for row in reader:
            writer.writerow([cell.replace(target, replacement) for cell in row])

    print(f"Done! Replaced {target} with {replacement}")


def interactive_mode():
    """Interactive processing mode."""
    print("CSV Path Replacement Tool")
    print("-" * 40)

    # Get input file
    while True:
        input_file = input("Enter CSV file path: ").strip()
        if os.path.exists(input_file):
            break
        print(f"Error: File {input_file} does not exist. Please try again.")

    # Confirm replacement rules
    print("\nReplacement Rules:")
    print(f"  '{'/mnt/windows/Datasets'}' → '{'/cloud/cloud-ssd1/dataset'}'")
    confirm = input("\nProceed with replacement? (y/n, default y): ").strip().lower()

    if confirm in ['', 'y', 'yes']:
        replace_dataset_path(input_file)
        print("\nProcessing finished!")
    else:
        print("Operation cancelled.")


def batch_scan_and_replace():
    """Scan current directory for CSV files and process them in bulk."""
    import glob

    # Get all CSV files in the current directory
    csv_files = glob.glob("*.csv") + glob.glob("*.CSV")

    if not csv_files:
        print("No CSV files found in the current directory.")
        return

    print(f"Found {len(csv_files)} CSV files:")
    for i, file in enumerate(csv_files, 1):
        print(f"  {i}. {file}")

    choice = input("\nProcess all files? (y/n, default y): ").strip().lower()

    if choice in ['', 'y', 'yes']:
        process_multiple_files(csv_files)
    else:
        # Let user choose specific files
        print("\nEnter file numbers to process (comma-separated, e.g., 1,3,4):")
        try:
            indices = input("Selection: ").strip()
            if indices:
                selected_indices = [int(i.strip()) - 1 for i in indices.split(',')]
                selected_files = [csv_files[i] for i in selected_indices if 0 <= i < len(csv_files)]
                process_multiple_files(selected_files)
        except ValueError:
            print("Invalid input format.")


# Simplified version optimized for specific dataset path migration
def quick_replace():
    """
    Quick replacement version - specifically handles /mnt/windows/Datasets → /cloud/cloud-ssd1/dataset.
    """
    import sys

    # Get command line argument or use default
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "annotations.csv"  # Default filename

    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist.")
        print("Usage: python script.py [input_file.csv]")
        return

    # Execute replacement
    target = "/mnt/windows/Datasets"
    replacement = "/cloud/cloud-ssd1/dataset"

    # Automatically generate output filename
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_cloud.csv"

    # Statistics
    changed_lines = 0
    total_lines = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', newline='', encoding='utf-8') as f_out:

        # Read line by line (supports multiple formats)
        for line in f_in:
            total_lines += 1

            # Check and replace target path
            if target in line:
                line = line.replace(target, replacement)
                changed_lines += 1

            f_out.write(line)

    print("✓ Quick replacement complete!")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")
    print(f"  Total lines: {total_lines}")
    print(f"  Modified lines: {changed_lines}")
    print(f"  Replacement: {target} → {replacement}")


if __name__ == "__main__":
    # Method 1: Direct call (Simplest)
    input_file = "/home/cam/LLM/vjepa2_deepspeed/datasets/imagenet_1k/imagenet_1k_train_path_shuffled.csv"
    output_file = "/home/cam/LLM/vjepa2_deepspeed/datasets/imagenet_1k/imagenet_1k_train_path_shuffled1.csv"
    replace_dataset_path(input_file, output_file)

    # Method 2: Quick version (CLI args)
    # quick_replace()

    # Method 3: Interactive mode
    # interactive_mode()

    # Method 4: Batch scan current directory
    # batch_scan_and_replace()

    # Method 5: Minimal one-liner
    # simple_one_liner()