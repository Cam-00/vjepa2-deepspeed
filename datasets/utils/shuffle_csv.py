import pandas as pd


def shuffle_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    df_shuffled.to_csv(output_file, index=False)


if __name__ == '__main__':
    input_file = '/home/cam/LLM/vjepa2_deepspeed/datasets/kinetics_700/kinetics_700_val_path.csv'
    output_file = '/home/cam/LLM/vjepa2_deepspeed/datasets/kinetics_700/kinetics_700_val_path_shuffled.csv'
    shuffle_csv(input_file, output_file)