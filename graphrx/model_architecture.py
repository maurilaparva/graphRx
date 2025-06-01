import torch
import torch.nn as nn
import torch.nn.functional as F

class SIGNEncoder(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_hops=2):
        super(SIGNEncoder, self).__init__()
        self.num_hops = num_hops
        self.linears = nn.ModuleList([nn.Linear(in_feats, hidden_feats) for _ in range(num_hops + 1)])
        self.out_layer = nn.Linear((num_hops + 1) * hidden_feats, out_feats)

    def forward(self, precomputed_feats):
        outs = []
        for i in range(self.num_hops + 1):
            h = torch.tanh(self.linears[i](precomputed_feats[i]))
            outs.append(h)
        h_cat = torch.cat(outs, dim=1)
        return F.leaky_relu(self.out_layer(h_cat))

class QuadraticDecoder(nn.Module):
    def __init__(self, embed_dim):
        super(QuadraticDecoder, self).__init__()
        self.scoring = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, drug_embs, disease_embs):
        scores = torch.sigmoid(torch.sum(drug_embs @ self.scoring * disease_embs, dim=1))
        return scores

class GDRnet(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_hops=2):
        super(GDRnet, self).__init__()
        self.encoder = SIGNEncoder(in_feats, hidden_feats, out_feats, num_hops)
        self.decoder = QuadraticDecoder(out_feats)

    def forward(self, precomputed_feats, drug_idx, disease_idx):
        embeddings = self.encoder(precomputed_feats)
        drug_embs = embeddings[drug_idx]
        disease_embs = embeddings[disease_idx]
        return self.decoder(drug_embs, disease_embs)
