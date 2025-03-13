import os
import csv
import numpy as np
import matplotlib.pyplot as plt


class Manager_Loss:
    def __init__(self, save_path="data/loss.csv"):
        """
        Manages storing and processing training loss values.
        - Stores loss per step.
        - Computes average loss per epoch.
        - Saves data to CSV.
        """
        self.save_path = save_path
        self.epoch_losses = []
        self.current_losses = []

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if not os.path.exists(save_path):
            with open(save_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["epoch", "average_loss"])  # CSV Header

    def store_loss(self, loss):
        """
        Store loss for the current training step.
        """
        self.current_losses.append(loss)

    def end_epoch(self, epoch):
        """
        Compute the average loss for the epoch and save to CSV.
        """
        if self.current_losses:
            avg_loss = np.mean(self.current_losses)
            self._save_to_csv(epoch, avg_loss)
            self.epoch_losses.append(avg_loss)
            self.current_losses = []  # Reset for next epoch

    def _save_to_csv(self, epoch, avg_loss):
        """
        Save the epoch's average loss to a CSV file.
        """
        with open(self.save_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch, avg_loss])
        print(f"Saved epoch {epoch} average loss: {avg_loss:.4f}")

    def load_losses(self):
        """
        Load saved loss values from the CSV file.
        """
        epochs, losses = [], []
        if os.path.exists(self.save_path):
            with open(self.save_path, mode="r") as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    epochs.append(int(row[0]))
                    losses.append(float(row[1]))
        return epochs, losses


if __name__ == "__main__":
    """
    If this file is run directly, it will plot the average loss over epochs.
    """
    manager_loss = Manager_Loss()
    epochs, losses = manager_loss.load_losses()

    if epochs and losses:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, losses, marker="o", linestyle="-", color="r", label="Average Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.title("Loss Over Training Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No loss data found.")
