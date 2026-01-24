from torch.utils.data import DataLoader
from dataset import LargeH5Dataset 
import multiprocessing



train_dataset = LargeH5Dataset(input_dir="./data/train_folder")

# Configuration du DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=multiprocessing.cpu_count()-1,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2  
)

print("Démarrage de l'entraînement...")

for epoch in range(10):
    for batch_data, batch_ids in train_loader:
        # batch_data est un Tensor (32, channels, longueur)
        # batch_data = batch_data.cuda()  # Envoi au GPU
        pass
