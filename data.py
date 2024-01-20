import torchaudio
import torch
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt

class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.file_list = self._create_file_list()
    
    def _create_file_list(self):
        file_list = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".wav"):
                    file_list.append(os.path.join(root, file))
        return file_list
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        return waveform, sample_rate

if __name__ == '__main__':
    dataset = AudioDataset(root_dir="Dataset_")
    
    waveform, sample_rate = dataset[0]
    print(f"Shape of waveform: {waveform.size()}")
    print(f"Sample rate of waveform: {sample_rate}")

    plt.figure()
    plt.plot(waveform.t().numpy())
    plt.show()
    