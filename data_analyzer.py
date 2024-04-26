import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder


ERC_DATASET_PATHS = {
    "train": "data/erc/MaSaC_train_erc.json",
    "val": "data/erc/MaSaC_val_erc.json",
    "test": "data/erc/MaSaC_test_erc.json"
}


def analyze_dataset(path):
    dataset_path = ERC_DATASET_PATHS[path]
    with open(dataset_path) as f:
        dataset = json.load(f)

    if dataset:
        first_entry = dataset[0]
        fields = list(first_entry.keys())
        print("Fields in the dataset:", fields)

        examples_in_dataset = len(dataset)
        print(f"Number of examples in {path} dataset: ", examples_in_dataset)
    else:
        print("The dataset is empty.")

    analyze_emotions(dataset, path)


def analyze_emotions(dataset, path):
    emotion_storage = {}
    for entry in dataset:
        for emotion in entry['emotions']:
            if emotion in emotion_storage:
                emotion_storage[emotion] += 1
            else:
                emotion_storage[emotion] = 1

    for emotion, count in emotion_storage.items():
        print(f"{emotion}: {count}")

    #plot_histogram(emotion_storage, path)
    check_correlation(dataset)


def plot_histogram(emotion_storage, path):
    emotions = list(emotion_storage.keys())
    counts = list(emotion_storage.values())

    plt.figure(figsize=(8, 6))
    plt.bar(emotions, counts, color='skyblue')
    plt.xlabel('Emotions')
    plt.ylabel('Occurrences')
    plt.tight_layout()
    plt.savefig(f'data/{path}_emotion_histogram.png')


def check_correlation(dataset):
    speakers = []
    emotions = []
    for entry in dataset:
        if 'speakers' in entry and 'emotions' in entry:
            speakers.extend(entry['speakers'])
            emotions.extend(entry['emotions'])

    le = LabelEncoder()

    x = le.fit_transform(speakers)
    y = le.fit_transform(emotions)

    r = np.corrcoef(x, y)

    print(r)


analyze_dataset("train")
