"""
nmf_speech_music_separation.py

Implements:
  - Supervised dictionary training per source (NMF with KL or IS divergence)
  - Mixture decomposition using concatenated dictionaries with combined KL+IS cost
    and temporal continuity regularizer on activations W.
  - Reconstruction with Wiener-like masks and filtering (median smoothing).

Dependencies:
  pip install numpy scipy librosa soundfile

Author: ChatGPT (reproduction/adaptation)
"""

import numpy as np
import librosa
import soundfile as sf
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
# Divergences and NMF updates for training B (multiplicative updates)
# -------------------------
def kl_divergence(V, R):
    # D_KL = sum(V * log(V/R) - V + R)
    R = np.maximum(R, 1e-12)
    V = np.maximum(V, 1e-12)
    return np.sum(V * np.log(V / R) - V + R)

def is_divergence(V, R):
    # D_IS = sum(V/R - log(V/R) - 1)
    R = np.maximum(R, 1e-12)
    V = np.maximum(V, 1e-12)
    return np.sum(V / R - np.log(V / R) - 1.0)

def nmf_train(V, rank, n_iter=200, divergence='kl', init_B=None, init_W=None, verbose=False):
    """
    Train NMF on V (F x T) for given divergence ('kl' or 'is').
    Returns B (F x rank), W (rank x T)
    """
    F, T = V.shape
    rng = np.random.default_rng(0)

    B = init_B.copy() if init_B is not None else np.maximum(1e-8, rng.random((F, rank)))
    W = init_W.copy() if init_W is not None else np.maximum(1e-8, rng.random((rank, T)))
    # normalize B columns
    B = B / (np.sum(B, axis=0, keepdims=True) + 1e-12)

    for it in range(n_iter):
        R = B.dot(W) + 1e-12

        if divergence == 'kl':
            # multiplicative updates KL
            # W <- W * (B^T (V/R)) / (B^T 1)
            numW = B.T.dot(V / R)
            denW = B.T.dot(np.ones_like(V))
            W *= numW / (denW + 1e-12)

            # B <- B * ((V/R) W^T) / (1 W^T)
            numB = (V / R).dot(W.T)
            denB = np.ones_like(V).dot(W.T)
            B *= numB / (denB + 1e-12)
        elif divergence == 'is':
            # multiplicative updates IS (see literature)
            # W <- W * (B^T ( (V / R**2) * R)) / (B^T (1/R))
            # Equivalent common formulation:
            numW = B.T.dot((V) / (R**2))
            denW = B.T.dot(1.0 / R)
            W *= np.sqrt((numW + 1e-12) / (denW + 1e-12))  # sqrt step improves stability
            numB = (V / (R**2)).dot(W.T)
            denB = (1.0 / R).dot(W.T)
            B *= np.sqrt((numB + 1e-12) / (denB + 1e-12))
        else:
            raise ValueError("divergence must be 'kl' or 'is'")

        # normalize B columns to avoid scale ambiguity
        scale = np.sum(B, axis=0, keepdims=True) + 1e-12
        B = B / scale
        W = W * scale  # compensate in W

        if verbose and (it % max(1, n_iter//10) == 0):
            if divergence == 'kl':
                obj = kl_divergence(V, B.dot(W))
            else:
                obj = is_divergence(V, B.dot(W))
            print(f"[train:{divergence}] iter {it}/{n_iter} obj {obj:.4e}")

    return B, W

# -------------------------
# Temporal smoothness gradient and utilities
# -------------------------
def temporal_gradient(W):
    """
    Compute gradient of temporal smoothness penalty:
      penalty = sum_t ||W[:,t] - W[:,t-1]||^2
    grad[:,t] = 2*(2W[:,t] - W[:,t-1] - W[:,t+1])
    with boundaries handled as one-sided differences.
    """
    K, T = W.shape
    grad = np.zeros_like(W)
    # interior
    if T >= 3:
        grad[:, 1:-1] = 2.0 * (2.0 * W[:, 1:-1] - W[:, :-2] - W[:, 2:])
    if T >= 2:
        # first column
        grad[:, 0] = 2.0 * (W[:, 0] - W[:, 1])  # derivative of (w0-w1)^2 wrt w0 is 2(w0-w1)
        # last column
        grad[:, -1] = 2.0 * (W[:, -1] - W[:, -2])
    return grad

# -------------------------
# Decomposition: fix B, estimate W minimizing
#   alpha KL + beta IS + lambda smoothness
# using projected gradient descent
# -------------------------
def estimate_activations(V, B, alpha=1.0, beta=1.0, lam=0.1,
                         n_iter=500, lr=1e-1, verbose=False):
    """
    V: F x T
    B: F x K (concatenated dictionaries)
    returns W: K x T (non-negative)
    Minimizes alpha*KL(V||BW) + beta*IS(V||BW) + lam * temporal_penalty(W)
    Using gradient descent (projected to >=0).
    """
    F, T = V.shape
    K = B.shape[1]
    rng = np.random.default_rng(1)
    W = np.maximum(1e-8, rng.random((K, T)))

    # precompute some
    one_FT = np.ones_like(V)

    for it in range(n_iter):
        R = B.dot(W) + 1e-12  # F x T
        # Gradients wrt R
        grad_R = np.zeros_like(R)
        if alpha != 0:
            # KL: dD/dR = -V/R + 1
            grad_R += alpha * (- V / R + 1.0)
        if beta != 0:
            # IS: dD/dR = -V / R^2 + 1 / R
            grad_R += beta * (- V / (R**2) + 1.0 / R)

        # chain rule: dD/dW = B^T (grad_R)
        grad_W = B.T.dot(grad_R)  # K x T

        # add temporal smoothness gradient (lambda * grad)
        if lam != 0:
            grad_W += lam * temporal_gradient(W)

        # gradient descent step (note signs: we add -lr * grad, because we wrote grad of objective)
        W = W - lr * grad_W

        # projection to non-negatives and small floor
        W = np.maximum(W, 1e-12)

        # optional small adaptive lr decay or norm check
        if verbose and (it % max(1, n_iter//8) == 0):
            # compute objective
            R = B.dot(W) + 1e-12
            obj = 0.0
            if alpha != 0:
                obj += alpha * kl_divergence(V, R)
            if beta != 0:
                obj += beta * is_divergence(V, R)
            if lam != 0:
                # temporal penalty
                diff = W[:, 1:] - W[:, :-1]
                obj += lam * np.sum(diff**2)
            print(f"[W-est] iter {it}/{n_iter} obj {obj:.4e}")
    return W

# -------------------------
# Post-processing / filtering / masking utilities
# -------------------------
def wiener_masks_from_BW(B, W, source_splits, V, eps=1e-12):
    """
    Given B (F x K), W (K x T), and source_splits = [(k0,k1), (k1,k2), ...]
    or list of indices per source, compute soft masks and per-source spectrogram estimates.
    Returns list of estimated magnitude spectrograms for each source (F x T).
    """
    recon = B.dot(W) + eps
    masks = []
    estimates = []
    for inds in source_splits:
        # indices list for this source
        if isinstance(inds, tuple) or isinstance(inds, list):
            comp = np.sum(B[:, inds].dot(W[inds, :]), axis=0) if False else None  # placeholder
        # better compute per-source recon via selecting atoms
        B_s = B[:, inds]  # F x ks
        W_s = W[inds, :]  # ks x T
        recon_s = B_s.dot(W_s)  # F x T
        mask = recon_s / recon  # soft mask
        est = mask * V
        masks.append(mask)
        estimates.append(est)
    return estimates, masks

def median_filter_masks(masks, kernel_time=7, kernel_freq=1):
    """Apply median filtering along time for each mask (to smooth activations)."""
    filtered = []
    for m in masks:
        # medfilt expects 2D; use kernel shape (freq, time)
        kfreq = max(1, kernel_freq)
        ktime = max(1, kernel_time)
        # medfilt from scipy works with 2D; ensure odd kernel sizes
        if ktime % 2 == 0:
            ktime += 1
        if kfreq % 2 == 0:
            kfreq += 1
        m_f = medfilt(m, kernel_size=(kfreq, ktime))
        filtered.append(np.maximum(m_f, 0.0))
    return filtered

# -------------------------
# Full pipeline wrapper
# -------------------------
def train_source_dictionaries(train_files, sr=16000, n_fft=1024, hop_length=None,
                              rank=8, divergence='kl', n_iter=200, verbose=False):
    """
    Train a dictionary B for a source given list of audio files.
    Returns: B (F x rank)
    """
    # Concatenate magnitude spectrograms across files along time axis
    mags = []
    for f in train_files:
        y, _ = librosa.load(f, sr=sr, mono=True)
        V, _, hop = stft_mag_phase(y, n_fft=n_fft, hop_length=hop_length)
        mags.append(V)
    Vbig = np.concatenate(mags, axis=1)  # F x Tbig
    print(f"[train] training {divergence} NMF on {Vbig.shape[1]} frames")
    B, W = nmf_train(Vbig, rank, n_iter=n_iter, divergence=divergence, verbose=verbose)
    return B

def separate_mixture(mixture_file, B_speech, B_music,
                     sr=16000, n_fft=1024, hop_length=None,
                     alpha=1.0, beta=1.0, lam=0.5, n_iter_W=400, lr=1e-1,
                     median_kernel_time=7, verbose=True, out_prefix="est"):
    """
    Complete separation pipeline:
      1) Load mixture
      2) Compute magnitude spectrogram V and phase
      3) Concatenate dictionaries and estimate W
      4) Compute soft masks, median-filter them, reconstruct signals
    """
    y_mix, _ = librosa.load(mixture_file, sr=sr, mono=True)
    V, phase, hop_length = stft_mag_phase(y_mix, n_fft=n_fft, hop_length=hop_length)
    F, T = V.shape
    # concatenate dictionaries
    B = np.concatenate([B_speech, B_music], axis=1)  # F x (K_s + K_m)
    K_total = B.shape[1]
    # index splits: speech indices [0:Ks], music [Ks:Ks+Km]
    Ks = B_speech.shape[1]
    Km = B_music.shape[1]
    source_splits = [list(range(0, Ks)), list(range(Ks, Ks + Km))]

    print("[sep] estimating activations W ...")
    W = estimate_activations(V, B, alpha=alpha, beta=beta, lam=lam,
                             n_iter=n_iter_W, lr=lr, verbose=verbose)

    # optional: smooth activations directly with uniform filter (time) to help
    W_smoothed = W.copy()
    for k in range(W.shape[0]):
        W_smoothed[k, :] = uniform_filter1d(W_smoothed[k, :], size=3, mode='reflect')

    # compute source estimates
    estimates_mag, masks = wiener_masks_from_BW(B, W_smoothed, source_splits, V)

    # median filter masks in time to reduce artifacts
    masks_filtered = median_filter_masks(masks, kernel_time=median_kernel_time, kernel_freq=1)

    # apply filtered masks and reconstruct time waveforms
    est_signals = []
    for i, m in enumerate(masks_filtered):
        mag_est = m * V
        y_est = istft_from_mag_phase(mag_est, phase, hop_length=hop_length)
        est_signals.append(y_est)
        # write out wav
        sf.write(f"{out_prefix}_source{i+1}.wav", y_est, sr)
        print(f"[sep] wrote {out_prefix}_source{i+1}.wav")

    return est_signals, masks_filtered, B, W

# -------------------------
# Example usage (script-like)
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--speech_files", nargs="+", help="speech training WAV files", required=False)
    parser.add_argument("--music_files", nargs="+", help="music training WAV files", required=False)
    parser.add_argument("--mixture", type=str, help="mixture wav file", required=False)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--rank_speech", type=int, default=8)
    parser.add_argument("--rank_music", type=int, default=12)
    parser.add_argument("--train_iter", type=int, default=200)
    parser.add_argument("--sep_iter", type=int, default=400)
    parser.add_argument("--alpha", type=float, default=1.0)  # KL weight
    parser.add_argument("--beta", type=float, default=1.0)   # IS weight
    parser.add_argument("--lam", type=float, default=0.5)    # temporal smoothness
    parser.add_argument("--out_prefix", type=str, default="est")
    args = parser.parse_args()

    if args.speech_files is None or args.music_files is None or args.mixture is None:
        print("Example usage: python nmf_speech_music_separation.py --speech_files s1.wav s2.wav --music_files m1.wav m2.wav --mixture mix.wav")
    else:
        B_s = train_source_dictionaries(args.speech_files, sr=args.sr, n_fft=args.n_fft,
                                        rank=args.rank_speech, divergence='kl', n_iter=args.train_iter, verbose=True)
        B_m = train_source_dictionaries(args.music_files, sr=args.sr, n_fft=args.n_fft,
                                        rank=args.rank_music, divergence='is', n_iter=args.train_iter, verbose=True)
        separate_mixture(args.mixture, B_s, B_m, sr=args.sr, n_fft=args.n_fft,
                         alpha=args.alpha, beta=args.beta, lam=args.lam, n_iter_W=args.sep_iter,
                         lr=1e-1, median_kernel_time=7, verbose=True, out_prefix=args.out_prefix)
