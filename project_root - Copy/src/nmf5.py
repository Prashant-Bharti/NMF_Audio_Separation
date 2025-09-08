import matplotlib.pyplot as plt

def plot_spectrogram(V, sr, hop_length, title, out_path):
    """Plot and save a log-magnitude spectrogram."""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(V, ref=np.max),
                             sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

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

    # Reconstruct time-domain signals + save audio
    output_paths = []
    for est, name in zip(estimates, ["speech", "music"]):
        S_est = est * np.exp(1j * phase)
        y_est = istft(S_est, hop_length=hop_length)
        out_path = os.path.join(out_dir, f"{name}_from_{os.path.basename(mixture_file)}")
        sf.write(out_path, y_est, sr)
        output_paths.append(out_path)

        # Save spectrogram as image
        plot_spectrogram(est, sr, hop_length, f"{name} spectrogram", out_path + ".png")

    # Also save mixture spectrogram
    plot_spectrogram(V, sr, hop_length, "Mixture spectrogram",
                     os.path.join(out_dir, f"mixture_{os.path.basename(mixture_file)}.png"))

    print(f"✅ Separated {mixture_file} → audio + spectrograms saved in {out_dir}")
