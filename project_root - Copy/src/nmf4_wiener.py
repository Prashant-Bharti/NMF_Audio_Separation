# Prashant Bharti
# 202251102
# To run this code:  
# python nmf4_wiener.py --speech_folder data/speech --music_folder data/music --mixture_folder data/mixture --out_dir results

import argparse
import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

EPS = 1e-10  # numerical stability


# -------------------------------
# Utility functions
# -------------------------------
def load_audio(path, sr=16000):
    """Load audio file, resample to sr, convert to mono."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y

def stft(y, n_fft=1024, hop_length=None):
    if hop_length is None:
        hop_length = n_fft // 4
    return librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

def istft(S, hop_length=None):
    return librosa.istft(S, hop_length=hop_length)

def magnitude_phase(S):
    return np.abs(S), np.angle(S)

def plot_spectrogram(V, sr, hop_length, title, out_path):
    """Plot and save spectrogram as image"""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        librosa.amplitude_to_db(V, ref=np.max),
        sr=sr,
        hop_length=hop_length,
        y_axis='log',
        x_axis='time'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# -------------------------------
# NMF training
# -------------------------------
def nmf_train(V, rank=20, n_iter=100):
    """Train NMF basis (dictionary) from magnitude spectrogram."""
    model = NMF(n_components=rank, init='random', max_iter=n_iter, solver='mu',
                beta_loss='kullback-leibler', random_state=0)
    W = model.fit_transform(V + EPS)
    H = model.components_
    return W, H

# -------------------------------
# Wiener filter
# -------------------------------
def wiener_filter(V, estimates, eps=EPS):
    """
    Apply Wiener filtering given mixture V and estimated sources.
    estimates: list of magnitude estimates (|S_i|)
    """
    estimates = np.stack(estimates, axis=0)  # shape: (n_sources, F, T)
    denom = np.sum(estimates, axis=0) + eps
    masks = estimates / denom
    return [masks[i] * V for i in range(len(estimates))]

# -------------------------------
# Separation
# -------------------------------
def separate_sources(mix_file, B_s, B_m, sr=16000, n_fft=1024, hop_length=None, out_dir="outputs", n_iter=200):
    """Perform source separation using fixed dictionaries B_s and B_m"""
    if hop_length is None:
        hop_length = n_fft // 4

    # Load mixture
    y, sr = librosa.load(mix_file, sr=sr)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    V, phase = magnitude_phase(S)

    # Stack dictionaries
    B = np.concatenate([B_s, B_m], axis=1)  # (F, K_total)
    K_s = B_s.shape[1]

    # Initialize activations
    H = np.abs(np.random.rand(B.shape[1], V.shape[1])) + EPS

    # Multiplicative updates (KL divergence)
    for it in range(n_iter):
        V_hat = B @ H + EPS
        H *= (B.T @ (V / V_hat)) / (B.T.sum(axis=1)[:, None] + EPS)

    # Reconstruct estimates
    speech_est = B[:, :K_s] @ H[:K_s, :]
    music_est = B[:, K_s:] @ H[K_s:, :]

    # Apply Wiener filtering
    estimates = wiener_filter(V, [speech_est, music_est])

    os.makedirs(out_dir, exist_ok=True)

    # Reconstruct signals and save
    for est, name in zip(estimates, ["speech", "music"]):
        S_est = est * np.exp(1j * phase)
        y_est = istft(S_est, hop_length=hop_length)

        out_path = os.path.join(out_dir, f"{name}_from_{os.path.basename(mix_file).replace('.mp3', '.wav')}")
        sf.write(out_path, y_est, sr)

        plot_spectrogram(est, sr, hop_length, f"{name} spectrogram", out_path + ".png")

    # Save mixture spectrogram
    plot_spectrogram(V, sr, hop_length, "Mixture spectrogram",
                     os.path.join(out_dir, f"mixture_{os.path.basename(mix_file)}.png"))

    print(f"✅ Separated {mix_file} → audio + spectrograms saved in {out_dir}")


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speech_folder", type=str, required=True)
    parser.add_argument("--music_folder", type=str, required=True)
    parser.add_argument("--mixture_folder", type=str, required=True)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    # Collect files
    speech_files = [os.path.join(args.speech_folder, f) for f in os.listdir(args.speech_folder) if f.endswith(".mp3")]
    music_files = [os.path.join(args.music_folder, f) for f in os.listdir(args.music_folder) if f.endswith(".mp3")]
    mixture_files = [os.path.join(args.mixture_folder, f) for f in os.listdir(args.mixture_folder) if f.endswith(".mp3")]

    print(f"Found {len(speech_files)} speech, {len(music_files)} music, {len(mixture_files)} mixtures.")

    if not speech_files or not music_files or not mixture_files:
        raise RuntimeError("Please put .mp3 files in data/speech, data/music, data/mixture")

    # Train speech dictionary
    V_s_list = []
    for f in speech_files:
        y = load_audio(f, sr=args.sr)
        V, _ = magnitude_phase(stft(y, n_fft=args.n_fft))
        V_s_list.append(V)
    V_s = np.concatenate(V_s_list, axis=1)
    B_s, _ = nmf_train(V_s, rank=args.rank, n_iter=args.n_iter)

    # Train music dictionary
    V_m_list = []
    for f in music_files:
        y = load_audio(f, sr=args.sr)
        V, _ = magnitude_phase(stft(y, n_fft=args.n_fft))
        V_m_list.append(V)
    V_m = np.concatenate(V_m_list, axis=1)
    B_m, _ = nmf_train(V_m, rank=args.rank, n_iter=args.n_iter)

    # Separate mixtures
    for mix_file in mixture_files:
        separate_sources(mix_file, B_s, B_m, sr=args.sr, n_fft=args.n_fft, out_dir=args.out_dir, n_iter=args.n_iter)


if __name__ == "__main__":
    main()
