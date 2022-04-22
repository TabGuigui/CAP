import torch
import faiss
from tqdm import tqdm

def get_bank(model, loader, logger, device):
    z_list = []
    for x, label in tqdm(loader):
        x = x.to(device)
        a,z,f_map = model(x) # featuremap
        z_list.append(z)
    z_list = torch.vstack(z_list)
    z_list_norm = z_list.cpu().data.numpy().copy()
    faiss.normalize_L2(z_list_norm)
    return z_list, z_list_norm


def knn_score(z_list, test_set, n_neighbours):
    """
    Calculates the KNN distance
    """

    index = faiss.IndexFlatIP(z_list.shape[1])
    index.add(z_list)
    if len(test_set.shape) == 1:
        test_set = test_set.unsqueeze(0)
    test_set = test_set.cpu().data.numpy()
    faiss.normalize_L2(test_set)
    D, I = index.search(test_set, n_neighbours)
    return D,I