import Levenshtein
import torch    
import torch.nn as nn  
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import os 
import numpy as np
from encoder_decoder_model import ARNModel
from torch.utils.data import DataLoader, Dataset


save_dir = "/content/drive/MyDrive/rna_data"
q_former_embeddings_dir = f"{save_dir}/qformer_embeddings"
embeddings_dir = f"{save_dir}/embeddings"
labels_file = f"{save_dir}/rna_labels.txt"
sequences_file = f"{save_dir}/rna_sequences.txt"
embeddings_index_file = f"{save_dir}/rna_embeddings_index.txt"

LATENT_DIM = 40  # Dimension des embeddings latents (D)
NUM_QUERIES = 16  # Nombre de queries (K)
VOCAB_SIZE = 5  # A, U, G, C, <PAD>
BATCH_SIZE = 16
EPOCHS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUCLEOTIDE_TO_INDEX = {"A": 0, "U": 1, "G": 2, "C": 3}

model = ARNModel().to(DEVICE)

# Charger la liste des fichiers embeddings
with open(embeddings_index_file, "r") as f_emb_index:
    embedding_filenames = [line.strip() for line in f_emb_index]

idx_to_token = {0: 'A', 1: 'U', 2: 'G', 3: 'C'}

def indices_to_sequence(indices, idx_to_token):
    tokens = [idx_to_token[idx] for idx in indices if idx in idx_to_token and idx_to_token[idx] not in ["<PAD>", "<EOS>", "<SOS>", "<UNK>"]]
    return ''.join(tokens)

def evaluate_model(model, dataloader, device, idx_to_token):
    """
    Évalue le modèle sur un DataLoader en calculant le NLL normalisé et le NED.
    - model : le modèle entraîné
    - dataloader : itérateur fournissant des paires (token_embs, target_seq)
    - device : torch.device
    - idx_to_token : dictionnaire de conversion indice -> token
    """
    model.eval()
    total_nll = 0.0
    total_ned = 0.0
    total_samples = 0

    with torch.no_grad():
        for token_embs, target_seq in dataloader:
            token_embs = token_embs.to(device)  # shape: (B, L_emb, 640)
            target_seq = target_seq.to(device)  # shape: (B, L_seq)
            B, L_seq = target_seq.size()

            # Passage forward : le modèle génère les logits
            output_logits = model(token_embs, target_seq)  # shape: (B, L_seq, VOCAB_SIZE)
            # Calcul des log-probabilités
            log_probs = F.log_softmax(output_logits, dim=-1)

            # Calcul de la NLL pour chaque token (on ignore le padding, ici index 4)
            loss_tensor = F.nll_loss(
                log_probs.view(-1, VOCAB_SIZE),
                target_seq.view(-1),
                reduction='none'
            ).view(B, L_seq)
            
            # Création d'un masque pour ne considérer que les tokens non-padding
            mask = (target_seq != 4).float()
            seq_lengths = mask.sum(dim=1)  # Longueur effective de chaque séquence

            # NLL normalisé par séquence
            nll_per_seq = (loss_tensor * mask).sum(dim=1) / seq_lengths
            total_nll += nll_per_seq.sum().item()

            # Décodage glouton pour obtenir la séquence reconstruite
            predicted_indices = output_logits.argmax(dim=-1)  # shape: (B, L_seq)

            # Calcul de la NED pour chaque échantillon
            for i in range(B):
                # On récupère uniquement les tokens réels (non-padding)
                length = int(seq_lengths[i].item())
                true_seq_indices = target_seq[i, :length].cpu().tolist()
                pred_seq_indices = predicted_indices[i, :length].cpu().tolist()

                # Utilisation de la fonction indices_to_sequence modifiée
                true_str = indices_to_sequence(true_seq_indices, idx_to_token)
                pred_str = indices_to_sequence(pred_seq_indices, idx_to_token)

                # Calcul de la distance de Levenshtein
                edit_distance = Levenshtein.distance(true_str, pred_str)
                ned = edit_distance / max(len(true_str), 1)  # Normalisation par la longueur
                total_ned += ned

            total_samples += B

    avg_nll = total_nll / total_samples
    avg_ned = total_ned / total_samples
    return avg_nll, avg_ned

class RNADatasetLazy(Dataset):
    def __init__(self, embedding_filenames, embeddings_dir, sequences_file, nucleotide_to_index):
        self.embedding_filenames = embedding_filenames
        self.embeddings_dir = embeddings_dir
        self.nucleotide_to_index = nucleotide_to_index

        # Au lieu de charger toutes les séquences, on enregistre les offsets des lignes
        self.line_offsets = []
        self.sequences_file = sequences_file
        with open(sequences_file, "r") as f:
            pos = f.tell()
            line = f.readline()
            while line:
                self.line_offsets.append(pos)
                pos = f.tell()
                line = f.readline()

    def __len__(self):
        return len(self.embedding_filenames)

    def __getitem__(self, idx):
        # Charger l'embedding correspondant
        emb_file = self.embedding_filenames[idx]
        emb = np.load(os.path.join(self.embeddings_dir, emb_file))
        token_embs = torch.tensor(emb, dtype=torch.float32)

        # Charger la séquence associée en se positionnant dans le fichier
        with open(self.sequences_file, "r") as f:
            f.seek(self.line_offsets[idx])
            seq = f.readline().strip()
        seq_indices = [self.nucleotide_to_index.get(nuc, 4) for nuc in seq]
        target_seq = torch.tensor(seq_indices, dtype=torch.long)

        return token_embs, target_seq

def collate_fn(batch):
    token_embs_list, target_seq_list = zip(*batch)
    token_embs_padded = pad_sequence(token_embs_list, batch_first=True)
    target_seq_padded = pad_sequence(target_seq_list, batch_first=True, padding_value=4)
    return token_embs_padded, target_seq_padded

def compute_ned():
    # Utilisation du Dataset lazy pour l'évaluation
    eval_dataset = RNADatasetLazy(embedding_filenames, embeddings_dir, sequences_file, NUCLEOTIDE_TO_INDEX)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Évaluation du modèle
    avg_nll, avg_ned = evaluate_model(model, eval_dataloader, DEVICE, idx_to_token)
    print(f"NLL normalisé: {avg_nll:.4f}, NED: {avg_ned:.4f}")
