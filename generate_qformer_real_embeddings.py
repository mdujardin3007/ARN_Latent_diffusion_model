import os 
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from encoder_decoder_model import QFormer

save_dir = "/content/drive/MyDrive/rna_data"
q_former_embeddings_dir = f"{save_dir}/qformer_embeddings"
embeddings_dir = f"{save_dir}/embeddings"
labels_file = f"{save_dir}/rna_labels.txt"
sequences_file = f"{save_dir}/rna_sequences.txt"
embeddings_index_file = f"{save_dir}/rna_embeddings_index.txt"

# ==============================
#  HYPERPARAMÈTRES
# ==============================
LATENT_DIM = 40  # Dimension des embeddings latents (D)
NUM_QUERIES = 16  # Nombre de queries (K)
VOCAB_SIZE = 5  # A, U, G, C, <PAD>
BATCH_SIZE = 16
EPOCHS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_qformer_real_embeddings():
    os.makedirs(q_former_embeddings_dir, exist_ok=True)

    # ==============================
    # CHARGEMENT DU MODÈLE
    # ==============================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qformer = QFormer().to(DEVICE)  # Seulement l'encodeur QFormer
    qformer.load_state_dict(torch.load("arn_model.pth", map_location=DEVICE), strict=False)
    qformer.eval()

    # ==============================
    # PRÉPARATION DES MEMMAPS
    # ==============================
    LATENT_DIM = 40
    NUM_QUERIES = 16
    with open(embeddings_index_file, "r") as f_emb_index:
        embedding_filenames = [line.strip() for line in f_emb_index]

    num_sequences = len(embedding_filenames)
    print(f"Nombre total de séquences ARN chargées : {num_sequences}")
    embedding_shape = (num_sequences, NUM_QUERIES, LATENT_DIM)

    # Sauvegarde des embeddings complets
    qformer_embeddings = np.memmap(
        f'{save_dir}/qformer_embeddings.dat', dtype='float32', mode='w+', shape=embedding_shape
    )

    labels_file = f'{save_dir}/rna_labels.txt'

    labels = np.loadtxt(labels_file, dtype=str, delimiter=None, usecols=0)


    # ==============================
    # EXTRACTION ET SAUVEGARDE DES EMBEDDINGS
    # ==============================
    print("Extracting Q-Former embeddings...")

    idx = 0
    index = 0

    for i in tqdm(range(0, num_sequences, BATCH_SIZE)):
        batch_filenames = embedding_filenames[i:i+BATCH_SIZE]

        # Chargement des embeddings bruts
        batch_token_embs = []
        for emb_file in batch_filenames:
            emb = np.load(os.path.join(embeddings_dir, emb_file))
            emb_tensor = torch.tensor(emb, dtype=torch.float32, device=DEVICE)
            batch_token_embs.append(emb_tensor)

        # Gestion des longueurs variables
        batch_token_embs = pad_sequence(batch_token_embs, batch_first=True)

        # ==============================
        # PASSAGE DANS LE Q-FORMER
        # ==============================
        with torch.no_grad():
            latents = qformer(batch_token_embs)  # (B, NUM_QUERIES, LATENT_DIM)

        # Sauvegarde des embeddings complets
        latents_np = latents.cpu().numpy()
        qformer_embeddings[index:index+latents_np.shape[0], :, :] = latents_np

        # ==============================
        # SAUVEGARDE INDIVIDUELLE POUR CHAQUE ÉCHANTILLON
        # ==============================
        for j, emb in enumerate(latents_np):
            embedding_filename = f"qformer_embedding_{idx}.npy"
            np.save(os.path.join(q_former_embeddings_dir, embedding_filename), emb)

            idx += 1

        index += latents_np.shape[0]

    # ==============================
    # FLUSH POUR ENREGISTRER LES MEMMAPS
    # ==============================
    qformer_embeddings.flush()

    print("Extraction terminée !")
    print("Shape des embeddings complets :", qformer_embeddings.shape)
