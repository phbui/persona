import os
import csv
import numpy as np
import matplotlib.pyplot as plt


class Manager_Reward:
    def __init__(self, save_path="data/rewards.csv"):
        """
        Manages storing and processing rewards from RL episodes.
        - Stores rewards per episode.
        - Computes average reward per epoch.
        - Saves data to CSV.
        """
        self.save_path = save_path
        self.epoch_rewards = []
        self.current_rewards = []

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if not os.path.exists(save_path):
            with open(save_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["epoch", "average_reward"])  # CSV Header

    def store_reward(self, reward):
        """
        Store reward for the current episode.
        """
        self.current_rewards.append(reward)

    def end_episode(self):
        """
        Marks the end of an episode and prepares for the next one.
        """
        if self.current_rewards:
            self.epoch_rewards.append(np.mean(self.current_rewards))  # Store episode average
            self.current_rewards = []

    def end_epoch(self, epoch):
        """
        Compute the average reward for the epoch and save to CSV.
        """
        if self.epoch_rewards:
            avg_reward = np.mean(self.epoch_rewards)
            self._save_to_csv(epoch, avg_reward)
            self.epoch_rewards = []  # Reset for next epoch

    def _save_to_csv(self, epoch, avg_reward):
        """
        Save the epoch's average reward to a CSV file.
        """
        with open(self.save_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch, avg_reward])
        print(f"Saved epoch {epoch} average reward: {avg_reward}")

    def load_rewards(self):
        """
        Load saved rewards from the CSV file.
        """
        epochs, rewards = [], []
        if os.path.exists(self.save_path):
            with open(self.save_path, mode="r") as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    epochs.append(int(row[0]))
                    rewards.append(float(row[1]))
        return epochs, rewards


if __name__ == "__main__":
    """
    If this file is run directly, it will plot the average reward over epochs.
    """
    manager_reward = Manager_Reward()
    epochs, rewards = manager_reward.load_rewards()

    if epochs and rewards:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, rewards, marker="o", linestyle="-", color="b", label="Average Reward")
        plt.xlabel("Epoch")
        plt.ylabel("Average Reward")
        plt.title("Average Reward Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No reward data found.")
