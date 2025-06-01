import torch
import torch.nn as nn
import torch.optim as optim
from model_architecture import GDRnet
from sklearn.model_selection import train_test_split

def train_model(hg, drug_feats, disease_feats, positive_pairs, negative_pairs, labels, epochs=1000, lr=1e-3):
    """
    Train the GDRNet model on the preprocessed data.
    """
    model = GDRnet(in_feats=drug_feats[0].shape[1], hidden_feats=64, out_feats=32, num_hops=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    combined_feats = [torch.cat([drug_feats[i], disease_feats[i]], dim=0) for i in range(len(drug_feats))]
    disease_offset = drug_feats[0].shape[0]
    adjusted_disease_idx = disease_idx + disease_offset

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(combined_feats, drug_idx, adjusted_disease_idx)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = ((preds > 0.5) == labels).float().mean()
            if epoch % 50 == 0:
                print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    return model
