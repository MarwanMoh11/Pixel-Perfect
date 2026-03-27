"""
Plot Training Loss Curves from CSV logs.
Generates the "Line Graph: Training Loss (Generator vs. Discriminator) over epochs"
required by the project proposal.
"""
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

def load_log(csv_path):
    """Load a training_log.csv into a dict of lists."""
    data = {'epoch': [], 'G_total': [], 'G_l1': [], 'G_perceptual': [], 'G_edge': [], 'D_loss': []}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in data:
                data[key].append(float(row[key]))
    return data

def plot_curves():
    os.makedirs('outputs', exist_ok=True)
    
    custom_log = 'models/checkpoints/training_log.csv'
    
    if not os.path.exists(custom_log):
        print(f"ERROR: {custom_log} not found. Run training first.")
        return
    
    data = load_log(custom_log)
    epochs = data['epoch']

    # ---- Plot 1: Generator vs Discriminator Loss (required by proposal) ----
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs, data['G_total'], color='#2ecc71', linewidth=2, label='Generator Total Loss')
    ax1.plot(epochs, data['D_loss'], color='#e74c3c', linewidth=2, label='Discriminator Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss: Generator vs Discriminator', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/training_loss_curves.png', dpi=150)
    print("Saved 'outputs/training_loss_curves.png'")

    # ---- Plot 2: Generator Loss Components Breakdown ----
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(epochs, data['G_l1'], linewidth=2, label='L1 Pixel Loss', color='#3498db')
    ax2.plot(epochs, data['G_perceptual'], linewidth=2, label='LPIPS Perceptual Loss', color='#9b59b6')
    ax2.plot(epochs, data['G_edge'], linewidth=2, label='Edge-Aware Sharpness Loss', color='#e67e22')
    ax2.plot(epochs, data['G_total'], linewidth=2, label='Total Generator Loss', color='#2ecc71', linestyle='--')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Generator Loss Components Over Training', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/generator_loss_breakdown.png', dpi=150)
    print("Saved 'outputs/generator_loss_breakdown.png'")

    print("\nAll training curve plots saved to outputs/")

if __name__ == '__main__':
    plot_curves()
