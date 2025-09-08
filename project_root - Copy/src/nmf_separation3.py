"""
nmf_speech_music_separation.py

Implements:
  - Supervised dictionary training per source (NMF with KL or IS divergence)
  - Mixture decomposition using concatenated dictionaries with combined KL+IS cost
    and temporal continuity regularizer on activations W.
  - Reconstruction with Wiener-like masks and filtering (median smoothing).

Dependencies:
  pip install numpy scipy librosa soundfile
"""

import numpy as np
import librosa
import soundfile as sf
import os, glob, argparse

from scipy.signal import medfilt
from scipy.ndimage import uniform_filter1d

# -------------------------
# Utility: STFT helpers
# -------------------------
def stft_mag_phase(y, n_fft=1024, hop_length=None, window='hann'):
    if hop_length is None:
        hop_length = n_fft // 4
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
    return np.abs(S), np.angle(S), hop_length

def istft_from_mag_phase(mag, phase, hop_length, window='hann'):
    S = mag * np.exp(1j * phase)
    return librosa.istft(S, hop_length=hop_length, window=window)

# -------------------------
# Divergences and NMF updates
# -------------------------
def kl_divergence(V, R):
    R = np.maximum(R, 1e-12)
    V = np.maximum(V, 1e-12)
    return np.sum(V * np.log(V / R) - V + R)

def is_divergence(V, R):
    R = np.maximum(R, 1e-12)
    V = np.maximum(V, 1e-12)
    return np.sum(V / R - np.log(V / R) - 1.0)

def nmf_train(V, rank, n_iter=200, divergence='kl', init_B=None, init_W=None, verbose=False):
    F, T = V.shape
    rng = np.random.default_rng(0)

    B = init_B.copy() if init_B is not None else np.maximum(1e-8, rng.random((F, rank)))
    W = init_W.copy() if init_W is not None else np.maximum(1e-8, rng.random((rank, T)))
    B = B / (np.sum(B, axis=0, keepdims=True) + 1e-12)

    for it in range(n_iter):
        R = B.dot(W) + 1e-12

        if divergence == 'kl':
            numW = B.T.dot(V / R)
            denW = B.T.dot(np.ones_like(V))
            W *= numW / (denW + 1e-12)

            numB = (V / R).dot(W.T)
            denB = np.ones_like(V).dot(W.T)
            B *= numB / (denB + 1e-12)

        elif divergence == 'is':
            numW = B.T.dot((V) / (R**2))
            denW = B.T.dot(1.0 / R)
            W *= np.sqrt((numW + 1e-12) / (denW + 1e-12))
            numB = (V / (R**2)).dot(W.T)
            denB = (1.0 / R).dot(W.T)
            B *= np.sqrt((numB + 1e-12) / (denB + 1e-12))

        else:
            raise ValueError("divergence must be 'kl' or 'is'")

        # scale = np.sum(B, axis=0, keepdims=True) + 1e-12
        # B = B / scale
        # W = W * scale
        scale = np.sum(B, axis=0, keepdims=True)  # shape (1, rank)
        B = B / scale
        W = W * scale.T   # now scale is (rank, 1), matches W


        if verbose and (it % max(1, n_iter//10) == 0):
            if divergence == 'kl':
                obj = kl_divergence(V, B.dot(W))
            else:
                obj = is_divergence(V, B.dot(W))
            print(f"[train:{divergence}] iter {it}/{n_iter} obj {obj:.4e}")

    return B, W

# -------------------------
# Temporal smoothness
# -------------------------
def temporal_gradient(W):
    K, T = W.shape
    grad = np.zeros_like(W)
    if T >= 3:
        grad[:, 1:-1] = 2.0 * (2.0 * W[:, 1:-1] - W[:, :-2] - W[:, 2:])
    if T >= 2:
        grad[:, 0] = 2.0 * (W[:, 0] - W[:, 1])
        grad[:, -1] = 2.0 * (W[:, -1] - W[:, -2])
    return grad

# -------------------------
# Estimate activations W
# -------------------------
def estimate_activations(V, B, alpha=1.0, beta=1.0, lam=0.1,
                         n_iter=500, lr=1e-1, verbose=False):
    F, T = V.shape
    K = B.shape[1]
    rng = np.random.default_rng(1)
    W = np.maximum(1e-8, rng.random((K, T)))

    for it in range(n_iter):
        R = B.dot(W) + 1e-12
        grad_R = np.zeros_like(R)

        if alpha != 0:
            grad_R += alpha * (- V / R + 1.0)
        if beta != 0:
            grad_R += beta * (- V / (R**2) + 1.0 / R)

        grad_W = B.T.dot(grad_R)

        if lam != 0:
            grad_W += lam * temporal_gradient(W)

        W = np.maximum(W - lr * grad_W, 1e-12)

        if verbose and (it % max(1, n_iter//8) == 0):
            R = B.dot(W) + 1e-12
            obj = 0.0
            if alpha != 0:
                obj += alpha * kl_divergence(V, R)
            if beta != 0:
                obj += beta * is_divergence(V, R)
            if lam != 0:
                diff = W[:, 1:] - W[:, :-1]
                obj += lam * np.sum(diff**2)
            print(f"[W-est] iter {it}/{n_iter} obj {obj:.4e}")
    return W

# -------------------------
# Masks and filtering
# -------------------------
def wiener_masks_from_BW(B, W, source_splits, V, eps=1e-12):
    recon = B.dot(W) + eps
    estimates = []
    masks = []
    for inds in source_splits:
        B_s = B[:, inds]
        W_s = W[inds, :]
        recon_s = B_s.dot(W_s)
        mask = recon_s / recon
        est = mask * V
        estimates.append(est)
        masks.append(mask)
    return estimates, masks

def median_filter_masks(masks, kernel_time=7, kernel_freq=1):
    filtered = []
    for m in masks:
        kfreq = max(1, kernel_freq)
        ktime = max(1, kernel_time)
        if ktime % 2 == 0:
            ktime += 1
        if kfreq % 2 == 0:
            kfreq += 1
        m_f = medfilt(m, kernel_size=(kfreq, ktime))
        filtered.append(np.maximum(m_f, 0.0))
    return filtered

# -------------------------
# Training
# -------------------------
def train_source_dictionaries(train_files, sr=16000, n_fft=1024, hop_length=None,
                              rank=8, divergence='kl', n_iter=200, verbose=False):
    mags = []
    for f in train_files:
        y, _ = librosa.load(f, sr=sr, mono=True)
        V, _, hop = stft_mag_phase(y, n_fft=n_fft, hop_length=hop_length)
        mags.append(V)
    Vbig = np.concatenate(mags, axis=1)
    print(f"[train] training {divergence} NMF on {Vbig.shape[1]} frames")
    B, W = nmf_train(Vbig, rank, n_iter=n_iter, divergence=divergence, verbose=verbose)
    return B

# -------------------------
# Separation
# -------------------------
def separate_mixture(mixture_file, B_speech, B_music,
                     sr=16000, n_fft=1024, hop_length=None,
                     alpha=1.0, beta=1.0, lam=0.5, n_iter_W=400, lr=1e-1,
                     median_kernel_time=7, verbose=True, out_prefix="est"):
    y_mix, _ = librosa.load(mixture_file, sr=sr, mono=True)
    V, phase, hop_length = stft_mag_phase(y_mix, n_fft=n_fft, hop_length=hop_length)

    B = np.concatenate([B_speech, B_music], axis=1)
    Ks = B_speech.shape[1]
    Km = B_music.shape[1]
    source_splits = [list(range(0, Ks)), list(range(Ks, Ks + Km))]

    print("[sep] estimating activations W ...")
    W = estimate_activations(V, B, alpha=alpha, beta=beta, lam=lam,
                             n_iter=n_iter_W, lr=lr, verbose=verbose)

    for k in range(W.shape[0]):
        W[k, :] = uniform_filter1d(W[k, :], size=3, mode='reflect')

    estimates_mag, masks = wiener_masks_from_BW(B, W, source_splits, V)
    masks_filtered = median_filter_masks(masks, kernel_time=median_kernel_time, kernel_freq=1)

    for i, m in enumerate(masks_filtered):
        mag_est = m * V
        y_est = istft_from_mag_phase(mag_est, phase, hop_length=hop_length)
        sf.write(f"{out_prefix}_source{i+1}.wav", y_est, sr)
        print(f"[sep] wrote {out_prefix}_source{i+1}.wav")

    return

# -------------------------
# File helpers
# -------------------------
def list_wav_files(folder):
    return sorted(glob.glob(os.path.join(folder, "*.wav")))

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speech_folder", type=str, default="data/speech")
    parser.add_argument("--music_folder", type=str, default="data/music")
    parser.add_argument("--mixture_folder", type=str, default="data/mixture")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--rank_speech", type=int, default=8)
    parser.add_argument("--rank_music", type=int, default=12)
    parser.add_argument("--train_iter", type=int, default=200)
    parser.add_argument("--sep_iter", type=int, default=400)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--out_prefix", type=str, default="est")
    args = parser.parse_args()

    speech_files  = list_wav_files(args.speech_folder)
    music_files   = list_wav_files(args.music_folder)
    mixture_files = list_wav_files(args.mixture_folder)

    print(f"Found {len(speech_files)} speech, {len(music_files)} music, {len(mixture_files)} mixtures.")

    if not speech_files or not music_files or not mixture_files:
        raise RuntimeError("Please put .wav files in data/speech, data/music, data/mixture")

    B_s = train_source_dictionaries(speech_files, sr=args.sr, n_fft=args.n_fft,
                                    rank=args.rank_speech, divergence='kl',
                                    n_iter=args.train_iter, verbose=True)
    B_m = train_source_dictionaries(music_files, sr=args.sr, n_fft=args.n_fft,
                                    rank=args.rank_music, divergence='is',
                                    n_iter=args.train_iter, verbose=True)

    for mix_file in mixture_files:
        mix_name = os.path.splitext(os.path.basename(mix_file))[0]
        out_prefix = f"{args.out_prefix}_{mix_name}"
        separate_mixture(mix_file, B_s, B_m, sr=args.sr, n_fft=args.n_fft,
                         alpha=args.alpha, beta=args.beta, lam=args.lam,
                         n_iter_W=args.sep_iter, lr=1e-1, median_kernel_time=7,
                         verbose=True, out_prefix=out_prefix)

if __name__ == "__main__":
    main()
