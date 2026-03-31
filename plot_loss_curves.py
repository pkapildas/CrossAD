import matplotlib.pyplot as plt
import re
import sys
import argparse
import os

def parse_and_plot(log_path, save_path="loss_curve.png"):
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}. Ensure you piped your terminal output to a file (e.g., > training.log)")
        sys.exit(1)

    epochs = []
    train_losses = []
    val_losses = []
    
    print(f"Parsing training trace from: {log_path}")
    
    with open(log_path, 'r') as file:
        lines = file.readlines()
        
    for line in lines:
        # Example: Epoch: 1, Steps: 100 | Train Loss: 0.123 Vali Loss: 0.050
        epoch_match = re.search(r'Epoch:\s*(\d+),\s*Steps:\s*\d+\s*\|\s*Train Loss:\s*([\d.]+)\s*Vali Loss:\s*([\d.]+)', line)
        if epoch_match:
            ep = int(epoch_match.group(1))
            tr_loss = float(epoch_match.group(2))
            va_loss = float(epoch_match.group(3))
            
            epochs.append(ep)
            train_losses.append(tr_loss)
            val_losses.append(va_loss)

    if not epochs:
        print("No valid 'Train Loss' / 'Vali Loss' signatures found in the log.")
        sys.exit(0)

    # Plot Configuration
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, train_losses, label="Combined Train Loss (MSE + Adv + Contrastive)", color='#E63946', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_losses, label="Validation Loss", color='#457B9D', linewidth=2, marker='s', markersize=4)

    plt.xlabel('Training Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Aggregate Graph Loss', fontsize=12, fontweight='bold')
    plt.title('CrossAD Multi-Objective Training Convergence Profiler', fontsize=16, pad=15)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="upper right", fontsize=10, shadow=True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    print(f"Loss Curve successfully plotted and saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CrossAD Loss Visualizer")
    parser.add_argument("--log", type=str, default="training.log", help="Path to your saved console output/log file.")
    parser.add_argument("--output", type=str, default="loss_curve.png", help="Where to save the PNG.")
    args = parser.parse_args()
    
    parse_and_plot(args.log, args.output)
