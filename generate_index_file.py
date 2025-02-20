import os 

save_dir = "/content/drive/MyDrive/rna_data"
embeddings_dir = f"{save_dir}/embeddings"
embeddings_index_file = f"{save_dir}/rna_embeddings_index.txt"


def generate_index_file():
    print("Generating embedding index file...")
    embedding_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.npy')]

    embedding_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    with open(embeddings_index_file, "w") as f_index:
        for file_name in embedding_files:
            f_index.write(file_name + "\n")

    print(f"Embedding index file saved at: {embeddings_index_file}")