# Speech Signal Denoising: Reducing High-Frequency Noise with Autoencoders


## Part 1: Conceptual Design

### Problem Statement
The goal of this project is to develop a speech signal denoiser capable of removing high-frequency noise from audio recordings. High-frequency noise—often introduced by environmental factors (e.g., electronic interference, wind) or recording equipment—degrades speech intelligibility and quality. Traditional denoising methods, such as filtering, often struggle with non-stationary or complex noise patterns and risk distorting the speech signal. To address this, we propose developing a deep learning-based speech denoiser using an autoencoder architecture. Autoencoders are well-suited for this task as they can learn to reconstruct clean signals from noisy inputs by capturing the underlying structure of the data.

### Proposed Solution
To address this challenge, the project will explore using a self-supervised autoencoder neural network trained on pairs of clean and synthetically corrupted speech signals. The denoiser will take a noisy speech signal as input and output a reconstructed, cleaner version. The model will be trained to reconstruct clean speech from synthetically noised input, using paired data without the need for manual labels.

At a high level, here are the key components of the proposed solution:
#### 1. **Autoencoder Architecture:**
We will follow the U-Net autoencoder architecture because it effectively captures both global and local patterns in structured data like spectrograms, making it well-suited for speech denoising tasks.

- **Encoder:** Compresses the noisy spectrogram into a low-dimensional latent representation using a series of convolutional and downsampling layers. This helps isolate core speech features while reducing noise.

- **Decoder:** Reconstructs a clean spectrogram from the latent representation using upsampling layers, guided by skip connections that restore fine-grained details lost during encoding.

We will experiment with different network depths, feature map sizes, and normalization techniques to balance model complexity and denoising performance.



#### 2. **Feature Representation:** 
The system will operate in the time-frequency domain.  Audio signals will be converted into spectrograms using a Short-Time Fourier Transform (STFT). This transforms the waveform into a structured 2D format (time vs frequency), making it easier to apply convolutional neural networks that can learn local and global noise patterns.

#### 3. **Learning Strategy:**
A hybrid loss function combining magnitude, waveform, and spectral reconstruction error guides the model to improve both objective and perceptual quality.

#### 4. **Noise Siimulation:**
Synthetic High-Frequency Noise: Clean speech from the LibriSpeech dataset will be corrupted with:

- **Gaussian Noise:** High-frequency band-limited noise.
- **Sine Waves:** Pure tones at frequencies >4 kHz.

### Dataset Requirements
#### Training Data:

- **Primary Source:** LibriSpeech train-clean-100 (100 hours of clean speech).

- **Noise Augmentation:** Synthetic high-frequency noise added to clean audio.

#### Validation Data:

- **Source:** LibriSpeech dev-clean (development set).

- **Purpose:** Tune hyperparameters (learning rate, network depth) and prevent overfitting.

#### Test Data:

- **Custom Recordings:** Real-world noisy speech samples recorded by the author.

- **Purpose:** Evaluate generalization to unseen noise types and recording conditions.

#### Key Technical Considerations
- **Phase Reconstruction:** Clean phase is not estimated; noisy phase is reused during inverse STFT. This is a simplification and may limit quality, but makes the system more stable and efficient.
- **Generalization:** While trained on synthetic noise, the final evaluation will include real-world noise to test robustness.


