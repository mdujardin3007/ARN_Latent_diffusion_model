import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from encoder_decoder_model import ARNModel

# ==============================
# HYPERPARAMÈTRES
# ==============================
LATENT_DIM = 40  # Dimension des embeddings latents (D)
NUM_QUERIES = 16  # Nombre de queries (K)
VOCAB_SIZE = 5  # A, U, G, C, <PAD>
BATCH_SIZE = 16
EPOCHS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dictionnaire de conversion nucléotide -> index
NUCLEOTIDE_TO_INDEX = {"A": 0, "U": 1, "G": 2, "C": 3}

# ==============================
# CHARGEMENT DES DONNÉES
# ==============================
save_dir = "/content/drive/MyDrive/rna_data"
embeddings_dir = f"{save_dir}/embeddings"
labels_file = f"{save_dir}/rna_labels.txt"
sequences_file = f"{save_dir}/rna_sequences.txt"
embeddings_index_file = f"{save_dir}/rna_embeddings_index.txt"

model = ARNModel().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=4)  # Ignore le padding


def train_encoder_decoder():
    # Charger la liste des fichiers embeddings
    with open(embeddings_index_file, "r") as f_emb_index:
        embedding_filenames = [line.strip() for line in f_emb_index]

    num_sequences = len(embedding_filenames)
    print(f"Nombre total de séquences ARN chargées : {num_sequences}")

    # ==============================
    # BOUCLE D'ENTRAÎNEMENT
    # ==============================
    for epoch in range(EPOCHS):
        total_loss = 0
        with open(sequences_file, "r") as f_seq:
            for i in tqdm(range(0, num_sequences, BATCH_SIZE)):
                batch_filenames = embedding_filenames[i:i+BATCH_SIZE]

                # Charger dynamiquement les embeddings pour chaque séquence : `(L, 640)`
                batch_token_embs = []
                for emb_file in batch_filenames:
                    emb = np.load(os.path.join(embeddings_dir, emb_file))
                    emb_tensor = torch.tensor(emb, dtype=torch.float32, device=DEVICE)
                    batch_token_embs.append(emb_tensor)

                # Gestion de longueurs variables via pad_sequence : `(B, max_L, 640)`
                batch_token_embs = pad_sequence(batch_token_embs, batch_first=True)

                # Lire les séquences ARN (une par échantillon)
                batch_rna_seqs = []
                for _ in range(len(batch_filenames)):
                    line = f_seq.readline().strip()
                    # Conversion des nucléotides en indices (4 est l'index pour le padding ou caractère inconnu)
                    indexed_seq = [NUCLEOTIDE_TO_INDEX.get(nuc, 4) for nuc in line]
                    batch_rna_seqs.append(torch.tensor(indexed_seq, dtype=torch.long, device=DEVICE))

                # Pad des séquences ARN pour avoir un batch de taille `(B, max_L_seq)`
                batch_rna_seqs = pad_sequence(batch_rna_seqs, batch_first=True, padding_value=4)

                optimizer.zero_grad()
                # On passe directement les embeddings bruts au modèle, qui appelle l'encodeur en interne.
                output = model(batch_token_embs, batch_rna_seqs)
                # Reshape pour le calcul de la perte
                loss = criterion(output.view(-1, VOCAB_SIZE), batch_rna_seqs.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / (num_sequences / BATCH_SIZE):.4f}")

    torch.save(model.state_dict(), "arn_model.pth")
    print("Entraînement terminé")
