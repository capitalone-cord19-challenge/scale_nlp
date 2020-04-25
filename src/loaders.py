import numpy as np
import torch

def rankingloader(features, batchsize):

    num_samples = len(features)
    idx = np.arrange(num_samples)

    for start in range(0, num_samples, batchsize):
        idxer = idx[start:start+batchsize]
        qid = []
        did = []
        dii = []
        dim = []
        dsi = []
        for i in range(idxer):
            qid.append(features[i].qid)
            did.append(features[i].did)
            dii.append(features[i].dii)
            dim.append(features[i].dim)
            dsi.append(features[i].dsi)
        dii = torch.tensor(dii, dtype=torch.long)
        dim = torch.tensor(dim, dtype=torch.long)
        dsi = torch.tensor(dsi, dtype=torch.lang)

        batch = (qid, did, dii, dim, dsi)
        yield batch
    return







