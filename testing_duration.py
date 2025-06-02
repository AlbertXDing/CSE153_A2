# %%
import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from tqdm import tqdm
import librosa
import numpy as np
import miditoolkit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score, accuracy_score
import random
import pretty_midi
import math
from symusic import Score
from miditok import REMI, TokenizerConfig
from midiutil import MIDIFile
from glob import glob
# used chatgpt to help me generate some functions

# %%
# Processing the midi files
midi_files = glob('nes_midis/*')
print(len(midi_files))

config = TokenizerConfig(num_velocities=1)
tokenizer = REMI(config)
tokenizer.train(vocab_size = 2000, files_paths=midi_files)


# %%
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
print("Torch CUDA version:", torch.version.cuda)
instruments = {}
bad_files = []

for file in midi_files:
    try:
        midi = pretty_midi.PrettyMIDI(file)
        for instrument in midi.instruments:
            name = pretty_midi.program_to_instrument_name(instrument.program)
            instruments[name] = instruments.get(name, 0) + 1
    except Exception as e:
        bad_files.append(file)

sorted_instruments = sorted(instruments.items(), key=lambda x: x[1], reverse=True)
midi_files = [file for file in midi_files if file not in bad_files]


# %%
# Using the top 20 instruments to condense the instrument types
useful_instruments = set(name for name, _ in sorted_instruments[:20]) 

# extracts only the notes where the instruments are useful
def extract_note_sequence(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                pitch = note.pitch
                duration = round(note.end - note.start, 3)  # Quantized duration
                notes.append((pitch, duration))
            
    return notes

#extract_notes(midi_files[1])

# %%

# class MIDIDataset(Dataset):
#     def __init__(self, midi_dir, vocab, seq_len=128):
#         self.data = []
#         self.vocab = vocab
#         self.seq_len = seq_len
#         self.pitch2idx = {p: i for i, p in enumerate(vocab)}

#         for file in midi_dir:
#             print(f"Processing file: {file}")
#             notes = extract_note_sequence(file)
#             encoded = [self.pitch2idx[n] for n in notes if n in self.pitch2idx]

#             for i in range(0, len(encoded) - seq_len):
#                 x = encoded[i:i + seq_len]
#                 y = encoded[i+seq_len]
#                 self.data.append((torch.tensor(x), torch.tensor(y)))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

class MIDIDataset(Dataset):
    def __init__(self, midi_dir, pitch_vocab, duration_vocab, seq_len=128):
        self.data = []
        self.pitch2idx = {p: i for i, p in enumerate(pitch_vocab)}
        self.dur2idx = {d: i for i, d in enumerate(duration_vocab)}
        self.seq_len = seq_len

        for file in midi_dir:
            notes = extract_note_sequence(file)
            encoded = [
                (self.pitch2idx[p], self.dur2idx[d])
                for (p, d) in notes if p in self.pitch2idx and d in self.dur2idx
            ]

            for i in range(len(encoded) - seq_len):
                x = encoded[i:i+seq_len]
                y = encoded[i+seq_len]
                pitches_x, durs_x = zip(*x)
                pitch_y, dur_y = y
                self.data.append((
                    torch.tensor(pitches_x), torch.tensor(durs_x),
                    torch.tensor(pitch_y), torch.tensor(dur_y)
                ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



# %%
pitch_vocab = sorted(set(p for f in midi_files for p, _ in extract_note_sequence(f)))
duration_vocab = sorted(set(d for f in midi_files for _, d in extract_note_sequence(f)))

pitch2idx = {p: i for i, p in enumerate(pitch_vocab)}
duration2idx = {d: i for i, d in enumerate(duration_vocab)}

# vocab = sorted(vocab)
dataset = MIDIDataset(midi_dir=midi_files, pitch_vocab=pitch_vocab, duration_vocab=duration_vocab)
print(f"Dataset size: {len(dataset)}") 
# loader = DataLoader(dataset[:1000], batch_size=32, shuffle=True, num_workers=0, pin_memory=True)


# %%
# class PitchLSTM(nn.Module):
#     def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2):
#         super().__init__()
#         self.embed = nn.Embedding(vocab_size, embed_dim)
#         self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, vocab_size)

#     def forward(self, x, hidden=None):
#         x = self.embed(x)
#         out, hidden = self.lstm(x, hidden)
#         out = self.fc(out[:, -1, :])  # use the last output for prediction
#         return out, hidden

class PitchDurLSTM(nn.Module):
    def __init__(self, pitch_vocab_size, dur_vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.pitch_embed = nn.Embedding(pitch_vocab_size, embed_dim)
        self.dur_embed = nn.Embedding(dur_vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim * 2, hidden_dim, batch_first=True)
        self.pitch_fc = nn.Linear(hidden_dim, pitch_vocab_size)
        self.dur_fc = nn.Linear(hidden_dim, dur_vocab_size)

    def forward(self, pitch_x, dur_x, hidden=None):
        p_emb = self.pitch_embed(pitch_x)
        d_emb = self.dur_embed(dur_x)
        x = torch.cat([p_emb, d_emb], dim=-1)  # shape (batch, seq_len, 2*embed_dim)
        out, hidden = self.lstm(x, hidden)
        last_out = out[:, -1, :]
        pitch_out = self.pitch_fc(last_out)
        dur_out = self.dur_fc(last_out)
        return pitch_out, dur_out, hidden


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
# model = PitchLSTM(vocab_size=len(vocab))
model = PitchDurLSTM(pitch_vocab_size=len(pitch_vocab), dur_vocab_size=len(duration_vocab))
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(15):
    # for batch_x, batch_y in loader:
    #     batch_x = batch_x.to(device)
    #     batch_y = batch_y.to(device)

    #     optimizer.zero_grad()
    #     out, _ = model(batch_x)
    #     loss = criterion(out, batch_y)
    #     loss.backward()
    #     optimizer.step()
    total_loss = 0
    num_batches = 0
    for pitch_x, dur_x, pitch_y, dur_y in loader:
        pitch_x, dur_x = pitch_x.to(device), dur_x.to(device)
        pitch_y, dur_y = pitch_y.to(device), dur_y.to(device)

        optimizer.zero_grad()
        pitch_out, dur_out, _ = model(pitch_x, dur_x)
        pitch_loss = criterion(pitch_out, pitch_y)
        dur_loss = criterion(dur_out, dur_y)
        loss = pitch_loss + dur_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)

    print(f"Epoch {epoch + 1} | Loss: {loss.item():.4f} | Perplexity: {perplexity:.2f}")






# %%
# def generate_sequence(model, start_seq, length, vocab_size, device):
#     model.eval()
#     generated = start_seq[:]

#     input_seq = torch.tensor(start_seq, dtype=torch.long).unsqueeze(0).to(device)  # shape (1, seq_len)

#     hidden = None
#     for _ in range(length):
#         with torch.no_grad():
#             out, hidden = model(input_seq, hidden)
#             prob = torch.softmax(out, dim=-1)
#             next_token = torch.multinomial(prob, num_samples=1).item()

#         generated.append(next_token)
#         input_seq = torch.tensor(generated[-len(start_seq):], dtype=torch.long).unsqueeze(0).to(device)

#     return generated

def generate_sequence(model, start_pitch_seq, start_dur_seq, length, device):
    model.eval()
    pitch_seq = start_pitch_seq[:]
    dur_seq = start_dur_seq[:]

    pitch_input = torch.tensor(start_pitch_seq, dtype=torch.long).unsqueeze(0).to(device)
    dur_input = torch.tensor(start_dur_seq, dtype=torch.long).unsqueeze(0).to(device)

    hidden = None
    for _ in range(length):
        with torch.no_grad():
            pitch_out, dur_out, hidden = model(pitch_input, dur_input, hidden)
            pitch_prob = torch.softmax(pitch_out, dim=-1)
            dur_prob = torch.softmax(dur_out, dim=-1)

            next_pitch = torch.multinomial(pitch_prob, num_samples=1).item()
            next_dur = torch.multinomial(dur_prob, num_samples=1).item()

        pitch_seq.append(next_pitch)
        dur_seq.append(next_dur)

        pitch_input = torch.tensor(pitch_seq[-len(start_pitch_seq):], dtype=torch.long).unsqueeze(0).to(device)
        dur_input = torch.tensor(dur_seq[-len(start_dur_seq):], dtype=torch.long).unsqueeze(0).to(device)

    return pitch_seq, dur_seq


# %%
# def sequence_to_midi(token_sequence, idx2pitch, output_path="generated.mid", velocity=100, duration=0.5):
#     midi = pretty_midi.PrettyMIDI()
#     instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

#     time = 0
#     for token in token_sequence:
#         pitch = idx2pitch[token]
#         duration = random.choice([0.25, 0.5, 1.0])
#         note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=time, end=time + duration)
#         instrument.notes.append(note)
#         time += duration

#     midi.instruments.append(instrument)
#     midi.write(output_path)

def sequence_to_midi(pitch_seq, dur_seq, idx2pitch, idx2dur, output_path="generated.mid", velocity=100):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    time = 0
    for p, d in zip(pitch_seq, dur_seq):
        pitch = idx2pitch[p]
        duration = idx2dur[d]
        note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=time, end=time + duration)
        instrument.notes.append(note)
        time += duration

    midi.instruments.append(instrument)
    midi.write(output_path)


# %%
# start_seq = [random.choice(range(len(vocab))) for _ in range(32)]
# generated_tokens = generate_sequence(model, start_seq, length=200, vocab_size=len(vocab), device=device)

# # Convert index to pitch
# idx2pitch = {i: p for i, p in enumerate(vocab)}
# sequence_to_midi(generated_tokens, idx2pitch, output_path="my_song.mid")

start_pitch_seq = [random.choice(range(len(pitch_vocab))) for _ in range(32)]
start_dur_seq = [random.choice(range(len(duration_vocab))) for _ in range(32)]

# seeding_choice = random.choice(midi_files)
# real_notes = extract_note_sequence(seeding_choice)
# print(seeding_choice)

# real_encoded = [
#     (pitch2idx[p], duration2idx[d])
#     for (p, d) in real_notes if p in pitch2idx and d in duration2idx
# ]

# start_pitch_seq, start_dur_seq = zip(*real_encoded[:32])
# start_pitch_seq = list(start_pitch_seq)
# start_dur_seq = list(start_dur_seq)

generated_pitches, generated_durs = generate_sequence(
    model, start_pitch_seq, start_dur_seq, length=200, device=device
)

idx2pitch = {i: p for i, p in enumerate(pitch_vocab)}
idx2dur = {i: d for i, d in enumerate(duration_vocab)}

sequence_to_midi(generated_pitches, generated_durs, idx2pitch, idx2dur, output_path="nes_song.mid")



