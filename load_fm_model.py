from google.colab import drive
import fm # install fm before running this script
import torch

drive.mount('/content/drive')
model_path = "/content/drive/MyDrive/RNA-FM_pretrained.pth"

def get_fm_model(model_path = model_path):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(f'using {device} device')

    fm_model, alphabet = fm.pretrained.rna_fm_t12(model_path)
    batch_converter = alphabet.get_batch_converter()
    fm_model.to(device)  # use GPU if available

    fm_model.eval()  # disables dropout for deterministic results
    return fm_model, batch_converter, alphabet, device