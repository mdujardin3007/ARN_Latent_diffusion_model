import random
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from load_fm_model import get_fm_model
from encoder_decoder_model import QFormer


fm_model, batch_converter, alphabet, device = get_fm_model()


# =============================================================================
# 1️⃣ Génération des ARN aléatoires
# =============================================================================

def sample_lengths(num_samples, min_len=50, max_len=800):
    """
    Échantillonne des longueurs de séquences selon une distribution log-normale
    tronquée pour approximer le pic vers ~100 et la longue traîne jusqu'à ~800.
    """
    lengths = []
    mu = 4.6   # e^4.6 ≈ 100
    sigma = 1.0
    while len(lengths) < num_samples:
        val = random.lognormvariate(mu, sigma)
        L = int(round(val))
        if min_len <= L <= max_len:
            lengths.append(L)
    return lengths

def generate_random_rna_sequences(num_samples):
    """
    Génère des séquences d'ARN aléatoires en choisissant la longueur selon la distribution
    définie et en tirant aléatoirement les nucléotides parmi {A, U, C, G}.
    """
    nucs = ['A', 'U', 'C', 'G']
    lengths = sample_lengths(num_samples)
    rna_sequences = []
    for L in lengths:
        seq = ''.join(random.choice(nucs) for _ in range(L))
        rna_sequences.append(seq)
    return rna_sequences, lengths

# Nombre de séquences à générer
num_samples = 2000
rna_sequences, lengths = generate_random_rna_sequences(num_samples)

# Visualisation de la distribution des longueurs
plt.figure(figsize=(8,5))
plt.hist(lengths, bins=50, density=True, alpha=0.7, color='tab:blue', edgecolor='black')
plt.title("Distribution des longueurs d'ARN générées")
plt.xlabel("Longueur de séquence")
plt.ylabel("Densité")
plt.show()

# Affichage d'un exemple de séquence générée
print("Exemple de séquence générée :", rna_sequences[0][:80], "...")
print("Longueur de cette séquence :", len(rna_sequences[0]))

# =============================================================================
# 2️⃣ Transformation des ARN en embeddings via RNA‑FM
# =============================================================================

# Répertoire de sauvegarde et fichiers
save_dir = "/content/drive/MyDrive/rna_data"

# Nombre de séquences (issu de rna_sequences)
num_sequences = len(rna_sequences)
print("Nombre de séquences :", num_sequences)

# Création d'un memmap pour stocker les embeddings de tokens
# (Forme attendue : (num_sequences, 1024, 640))
embedding_shape = (num_sequences, 1024, 640)
token_embeddings = np.memmap(os.path.join(save_dir, 'random_token_embeddings.dat'),
                             dtype='float32', mode='w+', shape=embedding_shape)

def generate_random_qformer_embeddings():

    # Chargement du modèle Q‑Former
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qformer = QFormer().to(DEVICE)  # Assurez-vous que QFormer est défini
    qformer.load_state_dict(torch.load("arn_model.pth", map_location=DEVICE), strict=False)
    qformer.eval()

    # On parcourt la liste des séquences aléatoires
    idx = 0   # Pour nommer les fichiers individuels
    index = 0 # Pour remplir le memmap


    num_sequences = num_samples

    # Paramètres du Q‑Former
    LATENT_DIM = 40
    NUM_QUERIES = 16
    latent_shape = (num_sequences, NUM_QUERIES, LATENT_DIM)


    random_qformer_embeddings = np.memmap(os.path.join(save_dir, 'random_qformer_embeddings.dat'),
                                dtype='float32', mode='w+', shape=latent_shape)

    for i, rna_seq in enumerate(tqdm(rna_sequences, desc="RNA‑FM")):
        # Préparation de l'entrée pour RNA‑FM
        seq_id = f"rna_{i}"
        data = [(seq_id, rna_seq)]
        _, _, batch_tokens = batch_converter(data)

        with torch.no_grad():
            results = fm_model(batch_tokens.to(device), repr_layers=[12])
        emb = results['representations'][12].cpu().numpy().squeeze(0)

        # Créer une liste contenant l'embedding courant
        batch_token_embs = [torch.tensor(emb, dtype=torch.float32, device=DEVICE)]

        # Convertir la liste en tenseur avec padding si nécessaire
        batch_token_embs = pad_sequence(batch_token_embs, batch_first=True)

        with torch.no_grad():
            latents = qformer(batch_token_embs)
        latents_np = latents.cpu().numpy()

        # Sauvegarde dans le memmap ou autre traitement...


        # Sauvegarde dans le memmap
        random_qformer_embeddings[index:index+latents_np.shape[0], :, :] = latents_np
        index += 1

    # Finalisation de l'écriture sur disque
    random_qformer_embeddings.flush()

    print("✅ Extraction terminée !")
