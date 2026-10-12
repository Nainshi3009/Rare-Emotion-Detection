from datasets import load_dataset
from collections import Counter

print("Loading GoEmotions...")

dataset = load_dataset("google-research-datasets/go_emotions")

train_data = dataset["train"]

# Get emotion names
emotion_names = train_data.features["labels"].feature.names

# Count emotions
counter = Counter()

for example in train_data:
    for label in example["labels"]:
        counter[label] += 1

print("\nTotal training examples:", len(train_data))
print("Total emotion classes:", len(emotion_names))

print("\nEmotion distribution:")
print("-" * 45)

for label_id, count in sorted(counter.items(), key=lambda x: x[1]):
    print(f"{emotion_names[label_id]:25s} {count:5d}")