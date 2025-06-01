import random
import torch

def sample_negative_pairs(num_neg, num_drugs, num_diseases, existing_pairs):
    """
    Sample negative pairs to balance the dataset.
    """
    negatives = set()
    while len(negatives) < num_neg:
        d = random.randint(0, num_drugs - 1)
        s = random.randint(0, num_diseases - 1)
        if (d, s) not in existing_pairs:
            negatives.add((d, s))
    return list(negatives)

def train_test_split_data(pairs, labels):
    """
    Split the dataset into train and test sets.
    """
    return train_test_split(pairs, labels, test_size=0.2, random_state=42, stratify=labels)
