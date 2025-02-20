import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

# Charger la liste des fichiers embeddings
with open(embeddings_index_file, "r") as f_emb_index:
    embedding_filenames = [line.strip() for line in f_emb_index]

num_sequences = len(embedding_filenames)
print(f"Nombre total de séquences ARN chargées : {num_sequences}")

# ==============================
# Q-FORMER (ENCODEUR GÉRANT L LONGUEUR VARIABLE)
# ==============================
class QFormer(nn.Module):
    def __init__(self, input_dim=640, latent_dim=LATENT_DIM, num_queries=NUM_QUERIES):
        super(QFormer, self).__init__()
        # K queries aléatoires
        self.queries = nn.Parameter(torch.randn(num_queries, latent_dim))
        # Attention multi-têtes
        self.attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=4, batch_first=True)
        # Projection de `(B, L, 640) → (B, L, 40)`
        self.proj = nn.Linear(input_dim, latent_dim)

    def forward(self, token_embs):
        B, L, _ = token_embs.shape  # L est variable
        token_embs = self.proj(token_embs)  # `(B, L, 40)`
        # Étendre les queries pour chaque échantillon de la batch : `(B, K, 40)`
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        latent_repr, _ = self.attention(queries, token_embs, token_embs)  # `(B, K, 40)`
        return latent_repr

# ==============================
# PROJECTION EN SOFT PROMPTS POUR LE DÉCODEUR
# ==============================
class LatentToPrompt(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(LatentToPrompt, self).__init__()
        # Projection de `(B, K, 40) → (B, K, 40)`
        self.fc = nn.Linear(latent_dim, latent_dim)

    def forward(self, latents):
        return self.fc(latents)

# ==============================
# CAUSAL TRANSFORMER DÉCODEUR
# ==============================
class CausalTransformerDecoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, num_tokens=VOCAB_SIZE):
        super(CausalTransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(num_tokens, latent_dim)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=latent_dim, nhead=4, batch_first=True),
            num_layers=4
        )
        self.fc_out = nn.Linear(latent_dim, num_tokens)

    def forward(self, soft_prompts, target_seq):
        # target_seq : `(B, L_target)`
        tgt_emb = self.embedding(target_seq)  # `(B, L_target, 40)`
        # Utilisation des soft_prompts (de taille `(B, 16, 40)`) comme mémoire
        output = self.transformer(tgt_emb, soft_prompts)  # `(B, L_target, 40)`
        return self.fc_out(output)  # `(B, L_target, VOCAB_SIZE)`

# ==============================
# MODÈLE COMPLET
# ==============================
class ARNModel(nn.Module):
    def __init__(self):
        super(ARNModel, self).__init__()
        self.encoder = QFormer()
        self.latent_to_prompt = LatentToPrompt()
        self.decoder = CausalTransformerDecoder()

    def forward(self, token_embs, target_seq):
        # token_embs : `(B, L, 640)`
        latent_repr = self.encoder(token_embs)  # `(B, 16, 40)`
        soft_prompts = self.latent_to_prompt(latent_repr)  # `(B, 16, 40)`
        return self.decoder(soft_prompts, target_seq)

# ==============================
