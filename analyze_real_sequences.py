import matplotlib.pyplot as plt 
from collections import Counter
from tqdm import tqdm
import numpy as np
import os 

# Définition des chemins des fichiers
save_dir = "/content/drive/MyDrive/rna_data"
sequences_file = f"{save_dir}/rna_sequences.txt"
labels_file = f"{save_dir}/rna_labels.txt"
embeddings_dir = f"{save_dir}/embeddings"
embeddings_index_file = f"{save_dir}/rna_embeddings_index.txt"

# Analyse de la distribution des bases (A, U, G, C)
base_counts = {"A": 0, "U": 0, "G": 0, "C": 0}

def analyze_sequences():
    with open(sequences_file, "r") as f_seq:
        for seq in f_seq:
            seq = seq.strip()  # Nettoyer les espaces et sauts de ligne
            for base in seq:
                if base in base_counts:
                    base_counts[base] += 1

    print("Distribution des nucléotides :", base_counts)

    # Affichage de la distribution des bases
    plt.figure(figsize=(6, 4))
    plt.bar(base_counts.keys(), base_counts.values(), color=['blue', 'orange', 'green', 'red'])
    plt.xlabel("Nucléotides (A, U, G, C)")
    plt.ylabel("Fréquence")
    plt.title("Distribution des nucléotides dans les séquences ARN")
    plt.show()

    # Analyse de la distribution des catégories ARN
    category_counts = Counter()

    with open(labels_file, "r") as f_labels:
        for category in f_labels:
            category_counts[category.strip()] += 1

    print("Distribution des catégories ARN :", category_counts)

    # Affichage de la distribution des catégories ARN
    plt.figure(figsize=(10, 5))
    plt.bar(category_counts.keys(), category_counts.values(), color='purple')
    plt.xticks(rotation=90)
    plt.xlabel("Catégories d'ARN")
    plt.ylabel("Nombre d'occurrences")
    plt.title("Distribution des catégories ARN")
    plt.show()

    # Analyse des tailles des embeddings
    embedding_sizes = []

    with open(embeddings_index_file, "r") as f_emb_index:
        embedding_filenames = f_emb_index.read().splitlines()

    for emb_file in tqdm(embedding_filenames, desc="Analyzing Embedding Sizes"):
        emb = np.load(os.path.join(embeddings_dir, emb_file))
        embedding_sizes.append(emb.shape[0])  # Taille variable de l'embedding

    print(f"Nombre total d'embeddings analysés : {len(embedding_sizes)}")
    print(f"Taille moyenne des embeddings : {np.mean(embedding_sizes):.2f}")
    print(f"Taille max des embeddings : {np.max(embedding_sizes)}")
    print(f"Taille min des embeddings : {np.min(embedding_sizes)}")

    # Affichage de la distribution des tailles d'embeddings
    plt.figure(figsize=(8, 5))
    plt.hist(embedding_sizes, bins=30, color='teal', alpha=0.7, edgecolor='black')
    plt.xlabel("Taille des embeddings")
    plt.ylabel("Nombre d'occurrences")
    plt.title("Distribution des tailles d'embeddings RNA-FM")
    plt.show()
