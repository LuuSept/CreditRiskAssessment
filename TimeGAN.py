import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class TimeGAN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(TimeGAN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.embed_fc = nn.Linear(hidden_dim, hidden_dim)

        self.generator = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.gen_fc = nn.Linear(hidden_dim, input_dim)

        self.discriminator = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.disc_fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, z=None, mode='generate'):
        if mode == 'embed':
            out, _ = self.embedder(x)
            out = self.embed_fc(out[:, -1, :])
            return out
        elif mode == 'generate':
            if z is None:
                batch_size = x.size(0)
                z = torch.randn(batch_size, self.hidden_dim).to(x.device)
            out, _ = self.generator(z.unsqueeze(1).repeat(1, x.size(1), 1))
            out = self.gen_fc(out)
            return out
        elif mode == 'discriminate':
            out, _ = self.discriminator(x)
            out = self.disc_fc(out[:, -1, :])
            return torch.sigmoid(out)
        return None


def generate_synthetic_data(model, real_data, num_samples):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.hidden_dim).to(device)
        seq_len = real_data.size(1)
        synthetic_data = model(None, z, mode='generate')
    return synthetic_data.cpu().numpy()


def train_timegan(model, dataloader, epochs=1000, lr=0.001):
    device = next(model.parameters()).device
    optimizer_G = optim.Adam(list(model.generator.parameters()) + list(model.embedder.parameters()), lr=lr)
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=lr)
    criterion = nn.BCELoss()
    model.train()
    for epoch in range(epochs):
        for real_data in dataloader:
            real_data = real_data[0].to(device)
            batch_size = real_data.size(0)
            optimizer_D.zero_grad()
            real_labels = torch.ones(batch_size, 1).to(device)
            real_output = model(real_data, mode='discriminate')
            d_loss_real = criterion(real_output, real_labels)
            z = torch.randn(batch_size, model.hidden_dim).to(device)
            fake_data = model(None, z, mode='generate').detach()
            fake_labels = torch.zeros(batch_size, 1).to(device)
            fake_output = model(fake_data, mode='discriminate')
            d_loss_fake = criterion(fake_output, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, model.hidden_dim).to(device)
            fake_data = model(None, z, mode='generate')
            fake_output = model(fake_data, mode='discriminate')
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            optimizer_G.step()
        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
    return model


def mahalanobis_distance_filtering(X_real, X_virtual, threshold_multiplier=1.5):
    mean = np.mean(X_real, axis=0)
    cov = np.cov(X_real, rowvar=False) + np.eye(X_real.shape[1]) * 1e-6
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)
    real_distances = [np.sqrt((sample - mean).dot(inv_cov).dot((sample - mean).T)) for sample in X_real]
    threshold = np.median(real_distances) * threshold_multiplier
    filtered_samples = [sample for sample in X_virtual if
                        np.sqrt((sample - mean).dot(inv_cov).dot((sample - mean).T)) < threshold]
    return np.array(filtered_samples)

# if __name__ == "__main__":
#     torch.manual_seed(42)
#     np.random.seed(42)
#     n_classes = 3
#     seq_len = 24
#     n_features = 10
#     n_real_samples = 50
#     n_virtual_samples = 200
#     epochs = 100
#     data_dict = {}
#     for fault_type in range(n_classes):
#         real_data = data_dict[fault_type]
#         real_data_tensor = torch.FloatTensor(real_data)
#         dataloader = DataLoader(TensorDataset(real_data_tensor), batch_size=32, shuffle=True)
#         timegan = TimeGAN(input_dim=n_features, hidden_dim=64)
#         timegan = train_timegan(timegan, dataloader, epochs=epochs)
#         synthetic_data = generate_synthetic_data(timegan, real_data_tensor, n_virtual_samples)
#         real_data_reshaped = real_data.reshape(-1, n_features)
#         synthetic_data_reshaped = synthetic_data.reshape(-1, n_features)
#         filtered_data = mahalanobis_distance_filtering(real_data_reshaped, synthetic_data_reshaped)
#         filtered_data = filtered_data.reshape(-1, seq_len, n_features)
#         np.save(f"fault_type_{fault_type}_augmented.npy", filtered_data)