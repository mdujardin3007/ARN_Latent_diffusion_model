import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
import importlib.util
module_path = '/content/drive/MyDrive/diffusion_model.py'
spec = importlib.util.spec_from_file_location("diffusion_model", module_path)
diffusion_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diffusion_module)
generate_latent_vectors = diffusion_module.generate_latent_vectors

save_dir = "/content/drive/MyDrive/rna_data"
q_former_embeddings_dir = f"{save_dir}/qformer_embeddings"
embeddings_dir = f"{save_dir}/embeddings"
labels_file = f"{save_dir}/rna_labels.txt"
sequences_file = f"{save_dir}/rna_sequences.txt"
embeddings_index_file = f"{save_dir}/rna_embeddings_index.txt"


latent_dim = 640

num_generated = 2000
num_samples = 2000
num_sequences = 2000

def t_sne_real_vs_generated_vs_random():
    generated_latents = generate_latent_vectors()

    #tsne for visualisation

    qformer_embeddings = np.memmap(f'{save_dir}/qformer_embeddings.dat', dtype='float32', mode='r', shape=(num_sequences, latent_dim))

    embeddings_for_tsne = np.array(qformer_embeddings)  # Now in memory

    random_qformer_embeddings = np.memmap(f'{save_dir}/random_qformer_embeddings.dat', dtype='float32', mode='r', shape=(num_sequences, latent_dim))

    random_embeddings_for_tsne = np.array(random_qformer_embeddings)  # Now in memory

    latent_data_real = torch.tensor(embeddings_for_tsne, dtype=torch.float32)
    latent_data_random = torch.tensor(random_embeddings_for_tsne, dtype=torch.float32)

    all_latents = np.concatenate([latent_data_real.numpy(), latent_data_random.numpy(), generated_latents], axis=0)[-6000:]
    labels_vis = np.array([0] * num_sequences + [1] *num_generated + [2] *num_samples)[-6000:]

    print("Running t-SNE on the combined latent space...")
    tsne = TSNE(n_components=2)
    all_latents_2d = tsne.fit_transform(all_latents)

    plt.figure(figsize=(8, 8))
    plt.scatter(all_latents_2d[labels_vis == 0, 0], all_latents_2d[labels_vis == 0, 1],
                color='tab:blue', s=25, alpha=0.8, label='Real')
    plt.scatter(all_latents_2d[labels_vis == 1, 0], all_latents_2d[labels_vis == 1, 1],
                color='tab:green', s=25, alpha=0.8, label='Random')
    plt.scatter(all_latents_2d[labels_vis == 2, 0], all_latents_2d[labels_vis == 2, 1],
                color='tab:red', s=25, alpha=0.8, label='Generated')
    plt.legend(markerscale=3, loc='best', fontsize=12)
    plt.xticks([])
    plt.yticks([])
    plt.title("t-SNE of Real vs. Random vs. Generated Latent Representations")
    plt.show()
