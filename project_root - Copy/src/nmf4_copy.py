import argparse
import os
import numpy as np
import librosa
import soundfile as sf
from sklearn.decomposition import NMF

EPS = 1e-10  # numerical stability


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


def nmf_train(V, rank=20, n_iter=100):
    """Train NMF basis (dictionary) from magnitude spectrogram."""
    model = NMF(n_components=rank, init='random', max_iter=n_iter, solver='mu',
                beta_loss='kullback-leibler', random_state=0)
    W = model.fit_transform(V + EPS)
    H = model.components_
    return W, H


# Separation

def wiener_filter(V, estimates, eps=EPS):
    """
    Apply Wiener filtering given mixture V and estimated sources.
    estimates: list of magnitude estimates (|S_i|)
    """
    estimates = np.stack(estimates, axis=0)  # shape: (n_sources, F, T)
    denom = np.sum(estimates, axis=0) + eps
    masks = estimates / denom
    return [masks[i] * V for i in range(len(estimates))]

# def separate_sources(mixture_file, B_s, B_m, sr=16000, n_fft=1024, hop_length=None, out_dir="results"):
#     """Separate a mixture using trained dictionaries B_s (speech), B_m (music)."""
#     os.makedirs(out_dir, exist_ok=True)

#     # Load mixture
#     y = load_audio(mixture_file, sr=sr)
#     S = stft(y, n_fft=n_fft, hop_length=hop_length)
#     V, phase = magnitude_phase(S)

#     # Stack dictionaries
#     B = np.concatenate([B_s, B_m], axis=1)  # (F, K_total)

#     # NMF inference
#     model = NMF(n_components=B.shape[1], init='custom', max_iter=100, solver='mu',
#                 beta_loss='kullback-leibler', random_state=0)
#     W = model.fit_transform(V + EPS, W=np.abs(np.random.rand(V.shape[0], B.shape[1])) + EPS, H=B.T + EPS)
#     H = model.components_

#     # Reconstruct source estimates
#     V_hat = W @ H
#     V_hat = np.maximum(V_hat, EPS)

#     # Split into speech/music estimates
#     speech_est = W[:, :B_s.shape[1]] @ H[:B_s.shape[1], :]
#     music_est = W[:, B_s.shape[1]:] @ H[B_s.shape[1]:, :]

#     # Wiener filtering for cleaner separation
#     estimates = wiener_filter(V, [speech_est, music_est])

#     # Reconstruct time-domain signals
#     for est, name in zip(estimates, ["speech", "music"]):
#         S_est = est * np.exp(1j * phase)
#         y_est = istft(S_est, hop_length=hop_length)
#         sf.write(os.path.join(out_dir, f"{name}_from_{os.path.basename(mixture_file)}"), y_est, sr)

#     print(f"✅ Separated {mixture_file} → saved in {out_dir}")
def separate_sources(mixture_file, B_s, B_m, sr=16000, n_fft=1024, hop_length=None, out_dir="results", n_iter=200):
    """Separate a mixture using trained dictionaries B_s (speech), B_m (music)."""
    os.makedirs(out_dir, exist_ok=True)

    # Load mixture
    y = load_audio(mixture_file, sr=sr)
    S = stft(y, n_fft=n_fft, hop_length=hop_length)
    V, phase = magnitude_phase(S)

    # Stack dictionaries (fixed bases)
    B = np.concatenate([B_s, B_m], axis=1)  # (F, K_total)
    F, K = B.shape
    T = V.shape[1]

    # Initialize activations H randomly
    H = np.abs(np.random.rand(K, T)) + EPS

    # Multiplicative updates (KL divergence)
    for it in range(n_iter):
        V_hat = B @ H + EPS
        H *= (B.T @ (V / V_hat)) / (B.T.sum(axis=1)[:, None] + EPS)

    # Reconstruct source estimates
    speech_est = B[:, :B_s.shape[1]] @ H[:B_s.shape[1], :]
    music_est = B[:, B_s.shape[1]:] @ H[B_s.shape[1]:, :]

    # Wiener filtering for cleaner separation
    estimates = wiener_filter(V, [speech_est, music_est])

    # Reconstruct time-domain signals
    for est, name in zip(estimates, ["speech", "music"]):
        S_est = est * np.exp(1j * phase)
        y_est = istft(S_est, hop_length=hop_length)
        sf.write(os.path.join(out_dir, f"{name}_from_{os.path.basename(mixture_file)}"), y_est, sr)

    print(f"✅ Separated {mixture_file} → saved in {out_dir}")



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
    speech_files = [os.path.join(args.speech_folder, f) for f in os.listdir(args.speech_folder) if f.endswith(".wav")]
    music_files = [os.path.join(args.music_folder, f) for f in os.listdir(args.music_folder) if f.endswith(".wav")]
    mixture_files = [os.path.join(args.mixture_folder, f) for f in os.listdir(args.mixture_folder) if f.endswith(".wav")]

    print(f"Found {len(speech_files)} speech, {len(music_files)} music, {len(mixture_files)} mixtures.")

    if not speech_files or not music_files or not mixture_files:
        raise RuntimeError("Please put .wav files in data/speech, data/music, data/mixture")

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
        separate_sources(mix_file, B_s, B_m, sr=args.sr, n_fft=args.n_fft, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
