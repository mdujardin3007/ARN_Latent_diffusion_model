import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE  # for dimension reduction

save_dir = "/content/drive/MyDrive/rna_data"
q_former_embeddings_dir = f"{save_dir}/qformer_embeddings"
embeddings_dir = f"{save_dir}/embeddings"
labels_file = f"{save_dir}/rna_labels.txt"
sequences_file = f"{save_dir}/rna_sequences.txt"
embeddings_index_file = f"{save_dir}/rna_embeddings_index.txt"

rfam_dict = {
    "Novel": "novel transcript",
    "Antisense": "antisense RNA",
    "Long non-coding": "long non-coding RNA",
    "Divergent": "divergent transcript",
    "Host Gene": "host gene",
    "microRNA": "microRNA",
    "Overlapping": "overlapping transcript",
    "Small nucleolar RNA": "small nucleolar RNA",
    "Long intergenic": "long intergenic RNA",
    "Autre": "other",
}

# ✅ Charger la liste des fichiers embeddings
with open(embeddings_index_file, "r") as f_emb_index:
    embedding_filenames = [line.strip() for line in f_emb_index]

num_sequences = len(embedding_filenames)
print(f"Nombre total de séquences ARN chargées : {num_sequences}")

latent_dim = 640

qformer_embeddings = np.memmap(f'{save_dir}/qformer_embeddings.dat', dtype='float32', mode='r', shape=(num_sequences, latent_dim))

embeddings_for_tsne = np.array(qformer_embeddings)  # Now in memory

def t_sne_qformer_embeddings():

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_for_tsne)
    print("t-SNE output shape:", embeddings_2d.shape)

    # Plot the t-SNE results.
    colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple',
            'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    labels_file = f'{save_dir}/rna_labels.txt'

    labels = np.loadtxt(labels_file, dtype=str, delimiter=None, usecols=0)

    plt.figure(figsize=(8, 8))
    unique_labels = sorted(list(set(labels)))
    print("Unique labels:", unique_labels)
    for i, label in enumerate(unique_labels):
        indices = [j for j, lab in enumerate(labels) if lab == label]
        plt.scatter(
            embeddings_2d[indices, 0],
            embeddings_2d[indices, 1],
            color=colors[i % len(colors)],
            s=25,
            alpha=0.8,
            label=rfam_dict.get(label, label)
        )
    plt.legend(markerscale=3, loc='best', fontsize=12)
    plt.xticks([])
    plt.yticks([])
    plt.title("t-SNE of RNA sequences by category")
    plt.show()