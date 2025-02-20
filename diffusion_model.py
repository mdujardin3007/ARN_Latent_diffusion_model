import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

save_dir = "/content/drive/MyDrive/rna_data"
q_former_embeddings_dir = f"{save_dir}/qformer_embeddings"
embeddings_dir = f"{save_dir}/embeddings"
labels_file = f"{save_dir}/rna_labels.txt"
sequences_file = f"{save_dir}/rna_sequences.txt"
embeddings_index_file = f"{save_dir}/rna_embeddings_index.txt"

with open(embeddings_index_file, "r") as f_emb_index:
    embedding_filenames = [line.strip() for line in f_emb_index]

num_sequences = len(embedding_filenames)
latent_dim = 640

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


qformer_embeddings = np.memmap(f'{save_dir}/qformer_embeddings.dat', dtype='float32', mode='r', shape=(num_sequences, latent_dim))

# reduced_embeddings = np.memmap(f'{save_dir}/reduced_embeddings.dat', dtype='float32', mode='r', shape=(num_sequences, latent_dim))
# Convert to a PyTorch tensor
latent_data = torch.tensor(np.array(qformer_embeddings), dtype=torch.float32)

# Helper: sinusoidal timestep embeddings (similar to what’s used in Transformers)
def get_timestep_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal timestep embeddings.
    timesteps: Tensor of shape (batch_size,)
    Returns a tensor of shape (batch_size, embedding_dim)
    """
    half_dim = embedding_dim // 2
    device = timesteps.device  # Ensure all constants are created on the same device
    emb_scale = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb_scale)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

#build dataset
batch_size = 64
dataset = TensorDataset(latent_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


#parameters of the diffusion
T = 1000
beta_start = 1e-4
beta_end = 0.02
# Create betas and move them to device
betas = torch.linspace(beta_start, beta_end, T).to(device)
alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, dim=0).to(device)


#model architecture

# We use a simple MLP that takes as input the noisy latent vector and a time embedding.
class DiffusionModel(nn.Module):
    def __init__(self, latent_dim, time_emb_dim=128):
        super(DiffusionModel, self).__init__()
        self.time_emb_dim = time_emb_dim
        self.fc1 = nn.Linear(latent_dim + time_emb_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, latent_dim)
        self.act = nn.ReLU()

    def forward(self, x, t):
        """
        x: tensor of shape (batch_size, latent_dim) — the noisy latent vectors.
        t: tensor of shape (batch_size,) — timesteps.
        """
        # Create time embeddings and concatenate with x
        t_emb = get_timestep_embedding(t, self.time_emb_dim).to(x.device)  # shape: (B, time_emb_dim)
        h = torch.cat([x, t_emb], dim=1)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        out = self.fc3(h)
        return out
    
#training loop

def diffusion_model_training():
    model = DiffusionModel(latent_dim=latent_dim, time_emb_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 12  # Adjust as needed, ca n'augmente plus au delà malheuresement

    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            x0 = batch[0].to(device)  # original (clean) latent vectors, shape: (B, latent_dim)
            batch_size = x0.shape[0]

            # Sample a random timestep for each example in the batch (integers between 0 and T-1)
            t = torch.randint(0, T, (batch_size,), device=device).long()
            # Get the corresponding alpha_bar (shape: (B, 1))
            alpha_bar_t = alphas_bar[t].unsqueeze(1).to(device)

            # Sample noise from a standard normal distribution
            noise = torch.randn_like(x0)

            # Create the noisy latent vector at time t:
            # x_t = sqrt(alpha_bar_t)*x0 + sqrt(1 - alpha_bar_t)*noise
            x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

            # The model predicts the noise component from the noisy latent x_t
            pred_noise = model(x_t, t)

            # Loss: mean squared error between the predicted noise and the true noise
            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_size
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {epoch_loss:.6f}")

#sampling from the model

# The following implements a simple DDPM sampling procedure.
@torch.no_grad()
def sample_ddpm(model, sample_shape, T, betas, alphas, alphas_bar):
    """
    model: trained diffusion model
    sample_shape: (batch_size, latent_dim)
    T, betas, alphas, alphas_bar: diffusion schedule parameters
    Returns: samples (latent vectors) of shape sample_shape
    """
    model.eval()
    x = torch.randn(sample_shape, device=device)  # start from pure noise
    for t in reversed(range(T)):
        t_tensor = torch.full((sample_shape[0],), t, device=device, dtype=torch.long)
        pred_noise = model(x, t_tensor)
        beta_t = betas[t].to(device)
        alpha_t = alphas[t].to(device)
        alpha_bar_t = alphas_bar[t].to(device)
        # DDPM update:
        # x0_pred = (x - sqrt(1 - alpha_bar_t)*pred_noise) / sqrt(alpha_bar_t)
        # Then compute the mean for the posterior and add noise if t > 0.
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0
        # The reverse update rule (as in Ho et al.):
        x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise) + torch.sqrt(beta_t) * noise
    return x


def generate_latent_vectors():
    model = DiffusionModel(latent_dim=latent_dim, time_emb_dim=128).to(device)

    # Generate new latent vectors (for example, generate as many as in your training set)
    num_generated = 2000  # or fewer if you prefer
    generated_latents = sample_ddpm(model, (num_generated, latent_dim), T, betas, alphas, alphas_bar)
    generated_latents = generated_latents.cpu().numpy()
    return generated_latents