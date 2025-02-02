# dataset.py

from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch

from dataset.data_processing import load_data


def prepare_dataloader_with_split(config, val_split=0.1):
    dataset = AudioFacialDataset(config)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=AudioFacialDataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=AudioFacialDataset.collate_fn)

    return train_dataset, val_dataset, train_dataloader, val_dataloader

def prepare_dataloader(config):
    dataset = AudioFacialDataset(config)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=AudioFacialDataset.collate_fn)
    return dataset, dataloader

class AudioFacialDataset(Dataset):
    def __init__(self, config):
        self.root_dir = config['root_dir']
        self.sr = config['sr']
        self.frame_rate = config['frame_rate']
        self.micro_batch_size = config['micro_batch_size']
        self.examples = []
        self.processed_folders = set()

        raw_examples = load_data(self.root_dir, self.sr, self.processed_folders)
        
        self.examples = []
        for audio_features, facial_data in raw_examples:
            processed_examples = self.process_example(audio_features, facial_data)
            if processed_examples is not None:
                self.examples.extend(processed_examples)
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @staticmethod
    def collate_fn(batch):
        src_batch, trg_batch = zip(*batch)
        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
        trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=0)
        return src_batch, trg_batch

    def process_example(self, audio_features, facial_data):

        num_frames_facial = len(facial_data)
        num_frames_audio = len(audio_features)

        max_frames = max(num_frames_audio, num_frames_facial)

        examples = []
        for start in range(0, max_frames - self.micro_batch_size + 1):
            end = start + self.micro_batch_size
            
            audio_segment = np.zeros((self.micro_batch_size, audio_features.shape[1]))
            facial_segment = np.zeros((self.micro_batch_size, facial_data.shape[1]))
            
            audio_segment[:min(self.micro_batch_size, num_frames_audio - start)] = audio_features[start:end]
            facial_segment[:min(self.micro_batch_size, num_frames_facial - start)] = facial_data[start:end]

            examples.append((torch.tensor(audio_segment, dtype=torch.float32), torch.tensor(facial_segment, dtype=torch.float32)))

        if max_frames % self.micro_batch_size != 0:
            start = max_frames - self.micro_batch_size
            end = max_frames

            audio_segment = np.zeros((self.micro_batch_size, audio_features.shape[1]))
            facial_segment = np.zeros((self.micro_batch_size, facial_data.shape[1]))
            
            segment_audio = audio_features[start:end]
            segment_facial = facial_data[start:end]

            reflection_audio = np.flip(segment_audio, axis=0)
            reflection_facial = np.flip(segment_facial, axis=0)

            audio_segment[:len(segment_audio)] = segment_audio
            audio_segment[len(segment_audio):] = reflection_audio[:self.micro_batch_size - len(segment_audio)]

            facial_segment[:len(segment_facial)] = segment_facial
            facial_segment[len(segment_facial):] = reflection_facial[:self.micro_batch_size - len(segment_facial)]

            examples.append((torch.tensor(audio_segment, dtype=torch.float32), torch.tensor(facial_segment, dtype=torch.float32)))
        
        return examples
