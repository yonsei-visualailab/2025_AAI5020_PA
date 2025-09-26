import torch
import numpy as np
from tqdm import tqdm

def compute_embeddings(model, dataset, device, batch_size=32):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings, indices, paths = [], [], []

    model.eval()
    with torch.no_grad():
        for imgs, idxs, ps in tqdm(loader, desc="Extracting embeddings"):
            imgs = imgs.to(device)
            embs = model(imgs).cpu().numpy()
            embeddings.append(embs)
            indices.extend(idxs.numpy())
            paths.extend(ps)

    embeddings = np.vstack(embeddings)
    return embeddings, indices, paths

def evaluate_recall(model, query_ds, db_ds, utmQ, utmDb, posDistThr, device, recall_values=[1,5,10]):
    q_embs, _, _ = compute_embeddings(model, query_ds, device)
    db_embs, _, _ = compute_embeddings(model, db_ds, device)

    recalls = {k: 0 for k in recall_values}

    """
    Your Task) Complete the function in evaluate_recall().

    - Evaluates recall@K for an image retrieval model.
    - Steps:
        1. Extract embeddings for query and database datasets using the given model.
        2. For each query embedding:
            - Compute similarity (or distance) to all database embeddings.
            - Sort database entries by similarity.
        3. Determine ground-truth positives:
            - Any database image within `posDistThr` of the query in UTM space.
            - If none exist, fall back to the nearest database image.
        4. For each K in recall_values:
            - Check whether at least one positive is retrieved within the top-K results.
        5. Count successful queries and compute recall@K percentages.
    
    Returns:
        dict: Recall@K results where keys are K and values are recall counts.
    """

    # TODO:
    # 1. Loop over all query embeddings:
    #     - Compute distances or cosine similarity to database embeddings.
    #     - Identify positives using utmDb and posDistThr. => These come from the custom class `LoadDataset(Dataset)`
    #     - If no positives, use the closest database image as fallback.
    #     - Update recall counts if a positive is within top-K results.
    # 2. Print recall@K values as percentages.
    # 3. Return the recall dictionary.

    for k in recall_values:
        print(f"Recall@{k}: {recalls[k]}/??? = {recalls[k]/max(1,len(query_ds))*100:.2f}%")

    return recalls
