import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from glob import glob
from model import NetVLADModel
'''
Newly defined libraries should be listed here.
'''

class IndoorDataset(Dataset):
    def __init__(self, base_dir, transform=None, scenes=None):
        self.items = []
        if scenes is None:
            scenes = ["P001","P002","P003","P004"]
        for scene in scenes:
            img_dir = os.path.join(base_dir, scene, "images")
            pose_file = os.path.join(base_dir, scene, "pose.txt")
            images = sorted(glob(os.path.join(img_dir, "*.png")))
            poses = np.loadtxt(pose_file).reshape(-1, 7)
            n = min(len(images), len(poses))
            for idx in range(0, n):
                self.items.append((images[idx], poses[idx], scene, idx))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, pose, scene, frame_idx = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, pose.astype(np.float32), img_path, scene, frame_idx

@torch.no_grad()
def extract_embeddings(model, loader, device):
    all_vecs, all_meta = [], []
    n_batches = 0
    for imgs, poses, paths, scenes, idxs in loader:
        n_batches += 1
        imgs = imgs.to(device)
        vecs = model(imgs).cpu()
        all_vecs.append(vecs)
        for i in range(len(paths)):
            p = poses[i].numpy() if hasattr(poses[i], "numpy") else np.asarray(poses[i])
            all_meta.append({
                "path": paths[i],
                "pose": p,
                "scene": scenes[i],
                "idx": int(idxs[i]),
            })
    return torch.cat(all_vecs, dim=0), all_meta

def pose_error(pose_q, pose_db):
    p1 = np.asarray(pose_q, dtype=np.float64)
    p2 = np.asarray(pose_db, dtype=np.float64)

    t_err = float(np.linalg.norm(p1[:3] - p2[:3]))

    q1, q2 = p1[3:7].copy(), p2[3:7].copy()
    n1, n2 = np.linalg.norm(q1), np.linalg.norm(q2)
    if n1 == 0 or n2 == 0:
        r_err = np.nan
    else:
        q1 /= n1; q2 /= n2
        dot = np.clip(abs(np.dot(q1, q2)), -1.0, 1.0)
        r_err = float(2.0 * np.degrees(np.arccos(dot)))

    return t_err, r_err

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NetVLADModel(num_clusters=16).to(device)
    ckpt_path = "./checkpoints/netvlad_final.pth"

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    model.load_state_dict(checkpoint)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    base_dir = "./dataset/camera"

    full_dataset = IndoorDataset(base_dir, transform=transform, scenes=["P001","P002","P003","P004"])  #scenes=["P001","P002","P003","P004"] 평가에서 변경 
    n_total = len(full_dataset)

    n_query = max(1, int(0.2 * n_total))
    g = torch.Generator().manual_seed(42) #manual_seed(42) 평가에서 변경 
    perm = torch.randperm(n_total, generator=g).tolist()

    query_indices = perm[:n_query]
    index_indices = perm[n_query:]

    query_set = Subset(full_dataset, query_indices)
    index_set  = Subset(full_dataset, index_indices)
        
    print(f"Total: {n_total} | Query: {len(query_set)} | Index: {len(index_set)}")

    query_loader = DataLoader(query_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    index_loader = DataLoader(index_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    q_vecs, q_meta = extract_embeddings(model, query_loader, device)
    db_vecs, db_meta = extract_embeddings(model, index_loader, device)

    # -------------------------------
    # TASK4 : Camera Pose Estimation
    # -------------------------------
    # example code: vector top-1
    # with torch.no_grad():
    #     dist_mat = torch.cdist(q_vecs, db_vecs, p=2)
    #     best_dists, best_db_idxs = dist_mat.min(dim=1)

    # rows = []
    # for qi in range(q_vecs.shape[0]):
    #     q_i = q_meta[qi]
    #     db_i = db_meta[int(best_db_idxs[qi])]
    #     t_err, r_err = pose_error(q_i["pose"], db_i["pose"])
    #     rows.append({
    #         "q_path": q_i["path"],
    #         "db_path": db_i["path"],
    #         "d_embed": float(best_dists[qi]),
    #         "t_err": float(t_err),
    #         "r_err": float(r_err),
    #     })
    # -------------------------------

    with open("result_pose_eval.txt", "w") as f:
        f.write("Top-1 retrieval pose evaluation\n")
        f.write(f"(queries={len(rows)}, index={len(db_meta)})\n\n")
        for r in rows:
            f.write(f"Query: {r['q_path']}\n")
            f.write(f"Best Match: {r['db_path']}\n")
            f.write(f"Embed Dist: {r['d_embed']:.4f}\n")
            f.write(f"Translation Error: {r['t_err']:.4f} m, Rotation Error: {r['r_err']:.2f} deg\n\n")

    mean_t = float(np.nanmean([r["t_err"] for r in rows]))
    mean_r = float(np.nanmean([r["r_err"] for r in rows]))
    print(f"Mean Pose Error: {mean_t:.4f}")
    print(f"Mean Rotation Error: {mean_r:.2f} deg")