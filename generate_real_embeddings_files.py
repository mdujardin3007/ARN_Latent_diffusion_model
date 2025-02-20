from pathlib import Path
from load_fm_model import get_fm_model
import numpy as np 
from tqdm import tqdm
import os 
import torch
from Bio import SeqIO

category_mapping = {
    "novel transcript": "Novel",
    "long intergenic non-protein coding RNA": "Long non-coding",
    "microRNA": "microRNA",
    "antisense RNA": "Antisense",
    "host gene": "Host Gene",
    "overlapping transcript": "Overlapping",
    "small nucleolar RNA": "Small nucleolar RNA",
    "divergent transcript": "Divergent",
    "long intergenic": "Long intergenic",
    "regulatory RNA": "Regulatory",
    "other": "Autre",
}


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

def get_arn_category(description):
    """ Associe une description à une catégorie d'ARN en utilisant le dictionnaire de mapping. """
    for key, category in category_mapping.items():
        if key in description:
            return category
    return "Autre"  # Si aucun mot-clé ne correspond, classer en "Autre"

def dna_to_rna(seq):
    """ Convertit une séquence ADN (A, T, G, C) en ARN (A, U, G, C). """
    return seq.replace("T", "U")


# Generator to stream sequences without loading everything into RAM
def stream_sequences(fasta_paths, species_filter, list_arn_chosen, nb_sample):
    for fasta_path in fasta_paths:
        compteur_arn = {}
        for record in SeqIO.parse(fasta_path, 'fasta'):
            if species_filter in record.description and len(record.seq) <= 768:
                category = get_arn_category(record.description)
                if category in list_arn_chosen:
                    compteur_arn[category] = compteur_arn.get(category, 0) + 1
                    if compteur_arn[category] <= nb_sample:
                        yield record.id, dna_to_rna(str(record.seq)), category

def generate_files():
    # Initialize parameters
    list_arn_chosen = [category_mapping[key] for key in category_mapping.keys()]
    species_filter = "Homo sapiens"
    nb_sample = 1500  # Limit per file (or None to process all)

    # Load FASTA files
    fasta_paths = [r"/content/drive/MyDrive/Colab Notebooks/ensembl.fasta"]

    # Create directories for storage
    save_dir = "/content/drive/MyDrive/rna_data"
    embeddings_dir = f"{save_dir}/embeddings"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(embeddings_dir).mkdir(parents=True, exist_ok=True)

    labels_file = f"{save_dir}/rna_labels.txt"
    sequences_file = f"{save_dir}/rna_sequences.txt"
    embeddings_index_file = f"{save_dir}/rna_embeddings_index.txt"

    # Step 1: Count the total number of sequences
    print("Counting sequences...")
    num_sequences = sum(1 for _ in stream_sequences(fasta_paths, species_filter, list_arn_chosen, nb_sample))
    print("Number of sequences:", num_sequences)

    embedding_shape = (num_sequences, 1024, 640)
    token_embeddings = np.memmap(f'{save_dir}/token_embeddings.dat', dtype='float32', mode='w+', shape=embedding_shape)

    # Step 2: Process and save progressively
    print("Processing RNA sequences...")

    idx = 0  # Index to name embedding files
    index = 0

    fm_model, batch_converter, alphabet, device = get_fm_model()

    with open(labels_file, "w") as f_labels, open(sequences_file, "w") as f_seq, open(embeddings_index_file, "w") as f_emb_index:
        os.makedirs(embeddings_dir, exist_ok=True)
        for seq_record in tqdm(stream_sequences(fasta_paths, species_filter, list_arn_chosen, nb_sample)):
            seq_id, rna_seq, category = seq_record

            # Limit RNA sequence to 768 tokens max
            rna_seq = rna_seq[:768] if len(rna_seq) > 768 else rna_seq

            # Prepare RNA for RNA-FM
            data = [(seq_id, rna_seq)]
            _, _, batch_tokens = batch_converter(data)

            with torch.no_grad():
                results = fm_model(batch_tokens.to(device), repr_layers=[12])

            emb = results['representations'][12].cpu().numpy().squeeze(0)
            token_embeddings[index:index+1, :emb.shape[0], :] = emb
            index += 1

            # Save embedding as a unique .npy file
            embedding_filename = f"embedding_{idx}.npy"
            np.save(os.path.join(embeddings_dir, embedding_filename), emb)

            # Save embedding index
            f_emb_index.write(f"{embedding_filename}\n")

            # Save labels and sequences
            f_labels.write(f"{category}\n")
            f_seq.write(f"{rna_seq}\n")

            idx += 1  # Increment index


    token_embeddings.flush()
