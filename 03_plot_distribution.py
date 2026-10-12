from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt

print("Loading GoEmotions...")

dataset = load_dataset("google-research-datasets/go_emotions")
train_data = dataset["train"]

emotion_names = train_data.features["labels"].feature.names

counter = Counter()

for example in train_data:
    for label in example["labels"]:
        counter[label] += 1

# Sort from rarest to most frequent
sorted_emotions = sorted(
    counter.items(),
    key=lambda x: x[1]
)

names = [emotion_names[label_id] for label_id, count in sorted_emotions]
counts = [count for label_id, count in sorted_emotions]

plt.figure(figsize=(12, 7))
plt.barh(names, counts)

plt.xlabel("Number of Training Examples")
plt.ylabel("Emotion")
plt.title("GoEmotions Training Set: Emotion Distribution")

plt.tight_layout()

plt.savefig("goemotions_distribution.png", dpi=300)

plt.show()

print("\nGraph saved as goemotions_distribution.png")