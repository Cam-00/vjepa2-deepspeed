import pandas as pd
import os
from collections import Counter


def balanced_subsample_no_header(input_csv_path, output_csv_path, random_state=4):
    """
    Read a headerless CSV file and perform balanced subsampling based on label categories.

    Parameters:
    input_csv_path: Path to the input CSV file.
    output_csv_path: Path to the output CSV file.
    random_state: Random seed to ensure reproducibility.
    """

    # 1. Read CSV file（no header）
    try:
        df = pd.read_csv(input_csv_path, header=None, delimiter=" ")
        print(f"read csv file successfully: {input_csv_path}")
        print(f"data shape: {df.shape}")
        print(f"first 3 row data:\n{df.head(3)}")
    except Exception as e:
        print(f"failed read csv file: {e}")
        return None

    # 2. Check the number of columns
    if df.shape[1] < 2:
        print("Error: CSV file must have at least 2 columns !")
        return None

    # 3. Setting the name of columns（the first col is data path, the second col is label）
    path_col = 'file_path'
    label_col = 'label'

    # rename columns
    df.columns = [path_col, label_col] + [f'extra_{i}' for i in range(2, df.shape[1])]

    # only need first 2 cols（can be adjusted if need）
    df = df.iloc[:, :2]

    # 4. Count the number of samples per category
    label_counts = Counter(df[label_col])
    print("Original category distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  category {label}: {count} samples")

    # 5. Determine sampling size (based on the minority class count)
    min_samples = min(label_counts.values())
    print(f"\n number of each category: {min_samples}")

    # 6. Perform balanced sampling by category
    balanced_samples = []

    for label in label_counts.keys():
        # Retrieve all samples for the current category
        class_samples = df[df[label_col] == label]

        # down sampling
        if len(class_samples) > min_samples:
            sampled_class = class_samples.sample(n=min_samples, random_state=random_state, replace=False)
        else:
            sampled_class = class_samples
            print(f"warning: category {label}'s samples are insufficient ，save all {len(class_samples)} samples")

        balanced_samples.append(sampled_class)

    # 7. concat all sampled samples
    balanced_df = pd.concat(balanced_samples, ignore_index=True)

    # shuffle the sampled data
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # 8. Statistics of the category distribution after sampling
    balanced_counts = Counter(balanced_df[label_col])
    print("\nClass distribution after balanced sampling:")
    for label, count in sorted(balanced_counts.items()):
        print(f"  Class {label}: {count} samples")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # 9. Save to a new CSV file (without header)
    try:
        balanced_df.to_csv(output_csv_path, index=False, header=False)
        print(f"\nBalanced sampling complete! Results saved to: {output_csv_path}")
        print(f"Sampled data shape: {balanced_df.shape}")

        # Display the first few rows of the sampled data
        print("\nFirst 3 rows of sampled data:")
        print(balanced_df.head(3).to_string(header=False, index=False))

        return balanced_df
    except Exception as e:
        print(f"Failed to save file: {e}")
        return None


# 更灵活的版本（支持自定义采样策略）
def flexible_balanced_sampling(input_csv_path, output_csv_path,
                               sampling_strategy='min', random_state=4):
    """
    Flexible balanced sampling function (for headerless CSV files).

    Args:
        input_csv_path: Path to the input CSV file.
        output_csv_path: Path to the output CSV file.
        sampling_strategy: Strategy for sampling.
            'min': Sample based on the minority class count (default).
            'max': Sample based on the majority class count.
            int: Fixed number of samples per category.
            dict: Custom sample counts per category, e.g., {0: 100, 1: 200}.
        random_state: Random seed for reproducibility.
    """

    # Read data (without header)
    df = pd.read_csv(input_csv_path, header=None, delimiter=" ")

    # Assuming the first column is the data path and the second is the label
    path_col = 0  # Index of the first column
    label_col = 1  # Index of the second column

    print(f"Data shape: {df.shape}")
    print(f"First 3 rows:\n{df.head(3).to_string(header=False, index=False)}")

    # Statistics of the original class distribution
    label_counts = Counter(df[label_col])
    print("\nOriginal class distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  Class {label}: {count} samples")

    # Determine sampling strategy
    if sampling_strategy == 'min':
        sample_size = min(label_counts.values())
        print(f"\nStrategy: Sample by minority class ({sample_size} samples/class)")
    elif sampling_strategy == 'max':
        sample_size = max(label_counts.values())
        print(f"\nStrategy: Sample by majority class ({sample_size} samples/class)")
    elif isinstance(sampling_strategy, int):
        sample_size = sampling_strategy
        print(f"\nStrategy: Fixed sampling ({sample_size} samples/class)")
    elif isinstance(sampling_strategy, dict):
        print(f"\nStrategy: Custom sampling {sampling_strategy}")
    else:
        sample_size = min(label_counts.values())
        print(f"\nStrategy: Default (Sample by minority class) ({sample_size} samples/class)")

    # Perform balanced sampling
    balanced_samples = []

    for label, count in sorted(label_counts.items()):
        # Retrieve all samples for the current category
        class_mask = df[label_col] == label
        class_data = df[class_mask]

        # Determine target sampling size
        if isinstance(sampling_strategy, dict):
            target_size = sampling_strategy.get(label, min(label_counts.values()))
        else:
            target_size = sample_size

        # Sampling logic
        if len(class_data) > target_size:
            # Downsampling
            sampled = class_data.sample(n=target_size, random_state=random_state, replace=False)
            print(f"  Class {label}: Downsampling {count} -> {target_size}")
        else:
            # # Oversampling (Sampling with replacement)
            # sampled = class_data.sample(n=target_size, random_state=random_state, replace=True)
            # print(f"  Class {label}: Oversampling {count} -> {target_size}")

            # If samples are insufficient, keep all available samples
            sampled = class_data
            print(f"Warning: Class {label} has insufficient samples, keeping all {len(class_data)} samples")

        balanced_samples.append(sampled)

    # Merge and shuffle data
    balanced_df = pd.concat(balanced_samples, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Save results (without header)
    balanced_df.to_csv(output_csv_path, sep=" ", index=False, header=False)

    # Final distribution statistics
    final_counts = Counter(balanced_df[label_col])
    print("\nFinal class distribution:")
    for label, count in sorted(final_counts.items()):
        print(f"  Class {label}: {count} samples")

    print(f"\nBalanced sampling complete!")
    print(f"Sampled data shape: {balanced_df.shape}")
    print(f"Results saved to: {output_csv_path}")
    print("\nFirst 3 rows of sampled data:")
    print(balanced_df.head(3).to_string(header=False, index=False))

    return balanced_df


# Usage Example
if __name__ == "__main__":
    # Input and output file paths
    input_file = "/home/cam/LLM/vjepa2_deepspeed/datasets/Something_Something_v2_deepspeed/ssv2_train_paths.csv"  # Replace with your input CSV path
    output_file = "/home/cam/LLM/vjepa2_deepspeed/datasets/ssv2_small_balanced_deepspeed/ssv2_train_paths_15_samples.csv"  # Output file path

    # # Method 1: Basic Balanced Sampling
    # print("=== Method 1: Basic Balanced Sampling ===")
    # result_df = balanced_subsample_no_header(input_file, output_file)
    #
    # print("\n" + "=" * 50 + "\n")

    # Method 2: Flexible Sampling Strategy
    print("=== Method 2: Flexible Sampling Strategy ===")
    # Choose one of the following strategies:

    # # Strategy 1: Sample by minority class (min)
    # result_df = flexible_balanced_sampling(
    #     input_file,
    #     "balanced_min.csv",
    #     sampling_strategy='min'
    # )

    # Strategy 2: Sample by majority class (max / oversampling)
    # result_df = flexible_balanced_sampling(
    #     input_file,
    #     "balanced_max.csv",
    #     sampling_strategy='max'
    # )

    # Strategy 3: Fixed quantity sampling
    result_df = flexible_balanced_sampling(
        input_file,
        output_file,
        sampling_strategy=100  # x samples per category
    )

    # Strategy 4: Custom sampling count per category
    # result_df = flexible_balanced_sampling(
    #     input_file,
    #     "balanced_custom.csv",
    #     sampling_strategy={0: 300, 1: 200, 2: 400}  # Custom counts
    # )