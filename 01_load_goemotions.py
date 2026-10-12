from datasets import load_dataset

print("Loading GoEmotions dataset...")

dataset = load_dataset("google-research-datasets/go_emotions")

print("\nDataset loaded successfully!")
print(dataset)

print("\nTraining set:")
print(dataset["train"][0])