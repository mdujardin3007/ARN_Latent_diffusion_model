import Levenshtein
from torch.utils.data import DataLoader, Dataset
from encoder_decoder_model import ARNModel
import importlib.util
module_path = '/content/drive/MyDrive/evaluate_model.py'
spec = importlib.util.spec_from_file_location("evaluate_model", module_path)
evaluate_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate_model)

RNADatasetLazy = evaluate_model.RNADatasetLazy
collate_fn = evaluate_model.collate_fn
indices_to_sequence = evaluate_model.indices_to_sequence
import torch
import os


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

# ✅ Charger la liste des fichiers embeddings
with open(embeddings_index_file, "r") as f_emb_index:
    embedding_filenames = [line.strip() for line in f_emb_index]

INDEX_TO_NUCLEOTIDE = {0: 'A', 1: 'U', 2: 'G', 3: 'C'}

eval_dataset = RNADatasetLazy(embedding_filenames, embeddings_dir, sequences_file, NUCLEOTIDE_TO_INDEX)
eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

def one_sequence_reconstructed():
    # On récupère un premier batch d'exemples depuis le DataLoader d'évaluation
    for token_embs, target_seq in eval_dataloader:
        token_embs = token_embs.to(DEVICE)
        target_seq = target_seq.to(DEVICE)
        break  # On ne prend qu'un batch

    # Passage dans le modèle pour obtenir la reconstruction (décodage glouton ici)
    with torch.no_grad():
        output_logits = model(token_embs, target_seq)
        predicted_indices = output_logits.argmax(dim=-1)  # Choix du token le plus probable pour chaque position

    # On récupère le premier exemple du batch
    original_indices = target_seq[0].cpu().tolist()
    predicted_indices_example = predicted_indices[0].cpu().tolist()

    # Conversion des indices en chaîne de caractères
    original_seq = indices_to_sequence(original_indices, INDEX_TO_NUCLEOTIDE)
    predicted_seq = indices_to_sequence(predicted_indices_example, INDEX_TO_NUCLEOTIDE)

    # Calcul de la distance d'édition
    edit_distance = Levenshtein.distance(original_seq, predicted_seq)

    # Affichage
    print("Séquence originale :", original_seq)
    print("Séquence reconstruite :", predicted_seq)
    print("Edit Distance :", edit_distance)
