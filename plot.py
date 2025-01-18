import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--file", type=str, required=True)

if __name__ == "__main__":
    # plt.style.use('seaborn')
    args = parser.parse_args()

    def parse_log_line(line):
        """Parse a single line of the log file into a dictionary"""
        # Split the line into parts
        parts = line.replace('train/', '').split(',')

        # First value is the iteration
        iteration = int(parts[0])

        # Parse the rest of the values
        values = {}
        for i in range(1, len(parts), 2):
            try:
                key = parts[i]
                value = float(parts[i+1])
                values[key] = value
            except:
                # Skip malformed entries
                continue

        values['iteration'] = iteration
        return values

# Read and parse the log file
    data = []
    # with open('v0.1.0/tae-v0.1.1-init-ldm-with-outlier/train.log', 'r') as f:
    with open(args.file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(parse_log_line(line))

# Convert to DataFrame
    df = pd.DataFrame(data)

# Create a figure with multiple subplots
    losses_to_plot = {
        'Loss Overview': ['total_loss'],
        'AE Loss': ['loss_ae'],
        'Disc Loss': ['loss_disc'],
        'Outlier Loss': ['outlier_loss'],
        'KL Loss': ['kl_loss'],
        'NLL Loss': ['nll_loss'],
    }

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Training Losses Over Time', fontsize=16)


    for (title, losses), ax in zip(losses_to_plot.items(), axes.flat):
        for i, loss in enumerate(losses):
            c = ["r", "b", "o", "k"][i]
            if loss in df.columns:
                ax.plot(df['iteration'], df[loss], label=loss, c=c)
        ax.set_title(title)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

# Adjust layout
    plt.tight_layout()
    plt.savefig('training_losses.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Plot saved as training_losses.png")
