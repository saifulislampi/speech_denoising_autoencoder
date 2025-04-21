import argparse
import json
import numpy as np
import torch
import librosa
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
from pesq import pesq

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.LeakyReLU(0.2),
                                  nn.Conv2d(32, 32, 3, padding=1), nn.LeakyReLU(0.2))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.LeakyReLU(0.2),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(0.2))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.bottleneck = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.2),
                                        nn.Conv2d(128, 128, 3, padding=1), nn.LeakyReLU(0.2))
        self.up2 = nn.Conv2d(128, 64, 1)
        self.dec2 = nn.Sequential(nn.Conv2d(64+64, 64, 3, padding=1), nn.LeakyReLU(0.2),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(0.2))
        self.up1 = nn.Conv2d(64, 32, 1)
        self.dec1 = nn.Sequential(nn.Conv2d(32+32, 32, 3, padding=1), nn.LeakyReLU(0.2),
                                  nn.Conv2d(32, 32, 3, padding=1), nn.LeakyReLU(0.2),
                                  nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        u2 = F.interpolate(self.up2(b), size=e2.shape[2:], mode='nearest')
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(self.up1(d2), size=e1.shape[2:], mode='nearest')
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return d1

def load_model_and_params(model_path, json_path, device="cuda"):
    with open(json_path) as f:
        params = json.load(f)
    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, params


def _denoise_segment(
    model: torch.nn.Module,
    params: dict,
    seg: np.ndarray,
    device = "cuda"
) -> np.ndarray:
    """Run one 2‑second segment (exactly 32000 samples) through the U‑Net."""
    gmin, gmax = params["global_min"], params["global_max"]
    # STFT → log‑mag → [0, 1]
    D = librosa.stft(seg,
                     n_fft=params["n_fft"],
                     hop_length=params["hop_length"],
                     win_length=params["win_length"])
    phase = np.angle(D)
    logmag = np.log1p(np.abs(D))
    logmag_norm = (logmag - gmin) / (1e-8 + (gmax - gmin))

    inp = torch.from_numpy(logmag_norm)[None, None].float().to(device)
    with torch.no_grad():
        pred_norm = model(inp)[0, 0].cpu().numpy()

    pred_logmag = pred_norm * (gmax - gmin) + gmin
    mag = np.expm1(pred_logmag)
    stft_pred = mag * np.exp(1j * phase)
    return librosa.istft(stft_pred,
                         hop_length=params["hop_length"],
                         win_length=params["win_length"],
                         length=len(seg))


def denoise_long_audio(
    model: torch.nn.Module,
    params: dict,
    noisy_path: str,
    out_path: str = None,
    seg_sec: float = 2.0,
    overlap: float = 0.5
) -> Tuple[np.ndarray, int]:
    """
    Denoise an audio file of any length using 2‑s overlap‑add.

    Returns (denoised_waveform, sample_rate).  Optionally writes to out_path.
    """
    sr = params["sr"]
    y, sr_loaded = librosa.load(noisy_path, sr=sr)
    if sr_loaded != sr:
        raise RuntimeError(f"Sample‑rate mismatch: loaded {sr_loaded}, expected {sr}")

    seg_len = int(seg_sec * sr)
    hop_len = int(seg_len * (1 - overlap))
    win = np.hanning(seg_len).astype(np.float32)

    # padded length so last segment is complete
    pad = (-(len(y) - seg_len) % hop_len) % hop_len
    y_padded = np.pad(y, (0, pad), mode="constant")

    out = np.zeros_like(y_padded, dtype=np.float32)
    acc = np.zeros_like(y_padded, dtype=np.float32)

    for start in range(0, len(y_padded) - seg_len + 1, hop_len):
        seg = y_padded[start:start + seg_len]
        den = _denoise_segment(model, params, seg)

        out[start:start + seg_len] += den * win
        acc[start:start + seg_len] += win

    # avoid division by tiny numbers
    eps = 1e-8
    denoised = out[:len(y_padded)] / (acc + eps)
    denoised = denoised[:len(y)]          # trim the padding

    if out_path:
        sf.write(out_path, denoised, sr)
        print(f"✓ Denoised file saved → {out_path}")

    return denoised, sr

def compute_snr(clean, denoised):
    noise = clean - denoised
    return 10 * np.log10(np.mean(clean**2) / (np.mean(noise**2) + 1e-10))

def main():
    parser = argparse.ArgumentParser(description="Denoise long audio files with SimpleUNet")
    parser.add_argument("noisy_audio", type=str, help="Path to noisy input audio (.flac/.wav)")
    parser.add_argument("--clean_audio", type=str, default=None, help="Path to clean reference audio")
    parser.add_argument("--output", type=str, default="denoised_output.flac", help="Output path")
    parser.add_argument("--model", type=str, default="model_2s_10_epoch.pth", help="Model file path")
    parser.add_argument("--params", type=str, default="model_2s_10_epoch_params.json", help="Params file path")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap ratio between segments")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, params = load_model_and_params(args.model, args.params, device)

    denoised_audio, sr = denoise_long_audio(
        model, params,
        noisy_path=args.noisy_audio,
        out_path=args.output,
        overlap=args.overlap
    )

    print(f"✓ Denoised audio saved to '{args.output}' at sample rate {sr}")

    if args.clean_audio:
        clean_audio, _ = librosa.load(args.clean_audio, sr=sr)
        min_len = min(len(clean_audio), len(denoised_audio))
        clean_audio, denoised_audio = clean_audio[:min_len], denoised_audio[:min_len]

        snr_value = compute_snr(clean_audio, denoised_audio)
        pesq_value = pesq(sr, clean_audio, denoised_audio, 'wb')

        print("\nEvaluation Metrics:")
        print(f"SNR:  {snr_value:.2f} dB")
        print(f"PESQ: {pesq_value:.2f}")

if __name__ == "__main__":
    main()
