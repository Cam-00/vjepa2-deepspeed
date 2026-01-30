import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import font_manager

# Set non-interactive backend
matplotlib.use('Agg')


# Find available Chinese fonts in the system
def find_chinese_font():
    """Find available Chinese fonts in the system"""
    # List of common Chinese fonts
    chinese_fonts = [
        'SimHei',  # HeiTi
        'Microsoft YaHei',  # Microsoft YaHei
        'SimSun',  # SimSun
        'KaiTi',  # KaiTi
        'FangSong',  # FangSong
        'STSong',  # STSong
        'Noto Sans CJK SC',  # Google Noto Font
        'WenQuanYi Micro Hei',  # WenQuanYi Micro Hei
        'Source Han Sans SC',  # Source Han Sans SC
    ]

    # Check which Chinese fonts are available in the system
    available_fonts = set([f.name for f in font_manager.fontManager.ttflist])
    found_fonts = [font for font in chinese_fonts if font in available_fonts]

    if found_fonts:
        print(f"Found available Chinese fonts: {found_fonts}")
        return found_fonts[0]  # Return the first font found
    else:
        print("No Chinese fonts found, using English labels")
        return None


# Set font
chinese_font = find_chinese_font()
if chinese_font:
    plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']
else:
    # If no Chinese font is available, use English labels
    print("Using English labels instead of Chinese")

plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')


def plot_two_columns(csv_file, epoch_col, ipe, x_col, y_col):
    """
    Read specific columns from a CSV file and plot the trend
    """
    try:
        # Only read the specified columns
        df = pd.read_csv(csv_file, usecols=[epoch_col, x_col, y_col])

        # Check if columns exist
        required_cols = [epoch_col, x_col, y_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return

        # Remove rows containing NaN values
        df_clean = df.dropna().copy()

        if len(df_clean) == 0:
            print("Warning: No valid data after cleaning")
            return

        # Calculate actual X-axis values
        df_clean['total_iterations'] = df_clean[epoch_col] * ipe + df_clean[x_col]

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot trend line
        ax.plot(df_clean['total_iterations'], df_clean[y_col],
                marker='o', linestyle='-', linewidth=1.5,
                markersize=3, alpha=0.8, color='steelblue')

        # Set labels and title
        ax.set_title(f'{y_col} vs Iterations Trend', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Total Iterations', fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')

        # Set Y-axis to log scale for loss metrics
        if 'loss' in y_col.lower():
            ax.set_yscale('log')
            ax.set_ylabel(f'{y_col} (log scale)', fontsize=12)

        # Add data statistics (in English)
        stats_text = f"""Data points: {len(df_clean)}
{y_col} range: {df_clean[y_col].min():.4f} - {df_clean[y_col].max():.4f}
Final value: {df_clean[y_col].iloc[-1]:.4f}"""

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)

        plt.tight_layout()

        # Save chart
        save_path = csv_file.replace('.csv', '_plot.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")

        plt.close(fig)

        # Display data information
        print("=" * 60)
        print("Data Statistics:")
        print(f"Data points: {len(df_clean)}")
        print(f"Iteration range: {df_clean['total_iterations'].min()} - {df_clean['total_iterations'].max()}")
        print(f"{y_col} range: {df_clean[y_col].min():.6f} - {df_clean[y_col].max():.6f}")
        print(f"{y_col} mean: {df_clean[y_col].mean():.6f}")
        print(f"{y_col} final value: {df_clean[y_col].iloc[-1]:.6f}")

    except FileNotFoundError:
        print(f"File not found: {csv_file}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function"""
    config = {
        'file_path': '/home/cam/LLM/vjepa2/output/pretrain/16.8.vitl.224px.16f/log_r0.csv',
        'epoch_col': 'epoch',
        'ipe': 300,
        'x_col': 'itr',
        'y_col': 'loss'
    }

    print("Starting to plot training curve...")
    plot_two_columns(
        csv_file=config['file_path'],
        epoch_col=config['epoch_col'],
        ipe=config['ipe'],
        x_col=config['x_col'],
        y_col=config['y_col']
    )


if __name__ == '__main__':
    main()