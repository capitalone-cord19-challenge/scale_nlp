import numpy as np
import torch

def rankingloader(features, batchsize):

    num_samples = len(features)
    idx = np.arange(num_samples)

    for start in range(0, num_samples, batchsize):
        batches = idx[start:start + batchsize]

        qid = [features[i].qid for i in batches]
        did = [features[i].did for i in batches]
        dii = torch.tensor([features[i].dii for i in batches], dtype=torch.long)
        dim = torch.tensor([features[i].dim for i in batches], dtype=torch.long)
        dsi = torch.tensor([features[i].dsi for i in batches], dtype=torch.long)
        batch = (qid, did, dii, dim, dsi)
        yield batch
    return







