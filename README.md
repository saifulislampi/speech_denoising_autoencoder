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



## Part 2: Dataset
This project uses the LibriSpeech ASR Corpus, a publicly available collection of clean, read English speech, as the primary source for training and validation data. To simulate noisy conditions, synthetic high-frequency noise will be added to clean speech samples, creating paired noisy-clean examples for training a self-supervised denoising model.

### Data Source

#### LibriSpeech (train-clean-100)

- ~100 hours of high-quality read English speech

- Used for generating training and validation data

- [Dataset Link](https://www.openslr.org/12)

#### LibriSpeech (dev-clean)

- Smaller development set for validation and tuning

- Ensures the model generalizes to unseen speaker data
- [Dataset Link](https://www.openslr.org/12)

#### Custom Noisy Speech Samples
- Real-world recordings of speech with high-frequency noise
- Used for final evaluation of the model's performance in practical scenarios

### Dataset Splits
- **Training Set (≈60%):** Subset of LibriSpeech train-clean-100, with synthetic noise added. Used to optimize model weights. And dev-clean for early prototyping.

- **Validation Set (≈20%):** Another subset from train-clean-100 and dev-clean. Used for hyperparameter tuning and early stopping.

- **Test Set (≈20%):** Another subset from train-clean-100 and dev-clean. And real-world recordings. Used for final evaluation of the model's performance.

### Data Preprocessing
Todo: Describe distributed data preprocessing steps using taskvine and floability here. Upload code. 

### Audio Samples
- **Clean Speech:** Original recordings from LibriSpeech. ([8555-292519-0015.flac](sample-data/clean/8555-292519-0015.flac))
- **Noisy Speech:** Clean speech samples with synthetic high-frequency noise added.(
[8555-292519-0015.flac](sample-data/noisy/8555-292519-0015.flac))

## Part 3: First Update

### Overview

In this stage of the project, we have begun implementing and testing our proposed speech denoiser based on a U-Net autoencoder. Our initial goals were:

1. To establish a workable training pipeline that can load clean/noisy speech pairs, split them into a consistent format, and feed them into our neural network.

2. To overcome dimension mismatches caused by variable-length audio.

3. To run enough epochs to obtain an initial sense of interim results on the test-clean subset of LibriSpeech (augmented with synthetic high-frequency noise).

We have now placed all relevant code in our GitHub repository, ensuring it is easy to follow. This code includes:

- Preprocessing scripts to split variable-length audio files into fixed 2-second segments.

- Dataset classes to load paired .flac files, compute STFT log-magnitudes, and apply the same normalization used during training.

- A U-Net model definition (in PyTorch) that downsamples along the time axis and uses skip connections to preserve high-resolution details.

- Training routines that combine a magnitude-domain loss (L1 on spectrogram) with a time-domain loss (L1 on the waveform reconstructed via the noisy phase).


### Architecture Recap
Our approach employs a U-Net autoencoder that operates on log-magnitude STFT features. In more detail:

#### 1. STFT & Log Magnitude
- Each audio file is sampled at 16 kHz.

- We compute the Short-Time Fourier Transform (STFT) using n_fft=512, hop_length=128, and win_length=512.

- We take the magnitude of each complex STFT frame, then apply log(1+magnitude). This compresses the dynamic range and helps the network learn more effectively.

#### 2. Min-Max Normalization
- We gather global min and max log-magnitudes across a portion of the training set to define a scaling from [0,1].

- We scale each spectrogram to [0,1], feed it to the network, and then unscale at the output stage. This helps keep training stable and consistent.

#### 3. U-Net Structure

- **Downsampling:** We use two downsampling levels, but only along the time dimension (via MaxPool2d(kernel_size=(1,2))). This preserves the frequency dimension (which is often 257 bins for n_fft=512) and prevents dimension mismatches when we do skip connections.

- **Bottleneck:** After the second downsampling, we have a “bottleneck” block that learns high-level or global features of the spectrogram.

- **Upsampling:** We upsample to match the skip-connection shapes using F.interpolate(..., size=...), ensuring the exact original frequency/time shape is recovered.

- **Final:** A Conv2d(..., out_channels=1) plus a sigmoid activation returns the predicted log-magnitude (normalized to [0..1]), which we then unscale and exponentiate to revert to a linear magnitude.

#### 4. Hybrid Loss

- **Spectrogram (magnitude) loss:** L1 difference between predicted log-magnitude and target log-magnitude.

- **Waveform loss:** We take our predicted magnitude, combine it with the noisy phase, do iSTFT, and then measure L1 difference in time domain.

- By weighting these two objectives, we encourage accurate magnitude prediction and a plausible waveform reconstruction.

<!-- 
Thus, the network can handle 2D spectrograms (time vs. frequency) with a consistent shape for each batch, while leveraging skip connections to capture both local and global features relevant to removing high-frequency noise. -->

### Challenges Encountered
#### 1. Variable-Length Audio
One of our biggest early hurdles was dealing with the fact that LibriSpeech test-clean files come in varying durations. Similarly, real-world recordings or augmented noisy files can also differ widely. If we simply load entire waveforms, we get random shape mismatches (e.g., the STFT might have a time dimension of 125 frames in one file and 121 frames in another).

- **Initial Attempt:** We tried random slicing or dynamic upsampling, but it complicated the skip connections in the U-Net.

- **Chosen Solution:** We decided to split all audio into 2-second segments (exactly 32,000 samples at 16 kHz). This ensures each segment has a consistent shape. Then, the U-Net always deals with a fixed frequency/time shape, no matter which segment is loaded.
 <!-- (time dimension = 250 STFT frames). -->
 

- **Implementation:** In the included notebook for part 1, you can see our create_2sec_segments(...) function that loads each (clean, noisy) file pair, cuts them into 2-second chunks, and saves them as .flac. We skip any leftover <2s to keep it simple. 

#### 2. Normalization
Another challenge was ensuring we store and re-use the same min,max from training time at inference. Without these exact values, the model sees a different input scale at inference and produces degraded results.

- We overcame this by saving the global_min and global_max in a JSON file (same name as the model). So, if you check out model_2s_10_epoch.pth you’ll also see model_2s_10_epoch_params.json in the repo.

- This ensures reproducibility and consistent inference on any new noisy file.

#### Matching Clean & Noisy
We also needed to guarantee that for each 2s chunk, the “clean” .flac file and the “noisy” .flac file truly align. Sorting file lists by name might not be reliable if folder structures differ.

- We overcame this by building “maps” keyed by basename. That is, for each .flac in the clean folder, we store a dictionary entry {filename -> full_path}, and do the same for the noisy folder. Then we intersect the sets to get a common set of filenames.

- This ensures the chunk “123-456-0001_seg0.flac” in the clean folder pairs exactly with “123-456-0001_seg0.flac” in the noisy folder.

### Training Speed & Preliminary Results
We trained for ~10 epochs on the LibriSpeech test-clean data. The results are promising: you can audibly hear a significant reduction in high-frequency hiss when comparing the “Noisy” vs. “Denoised” audio. However, some distortion or muffling may still exist. We plan to refine the model further with more data and more epochs.

- **Noisy 2 Test Sample:** [noisy-test-sample-1.wav](sample-data/noisy/noisy-test-sample-1.wav)
- **Denoised 2s Test Sample:** [denoised-test-sample-1.wav](sample-data/denoised/denoised-test-sample-1.wav)

### Next Steps
#### 1. Train on Larger Data
Our next major milestone is to move beyond the ~1–2 hours of test-clean data to the train-clean-100 set (100 hours). That means we’ll generate many more 2-second segments, leading to a dataset large enough to capture robust patterns and handle real-world high-frequency noise.

We anticipate better generalization and more stable training with a larger dataset.

#### 2. Explore Additional Loss Functions

We currently combine an L1 spectrogram loss with an L1 time-domain loss. However, we might also consider more advanced STFT losses that weigh different frequency regions.
We can also measure objective speech quality metrics like PESQ or STOI to track improvements.


#### 3. Evaluate on Unseen Audio
We want to test on “unknown” audio , potentially from real-world recordings with environmental or equipment-based high-frequency noise. Because we store and share global_min, global_max, and STFT parameters, we can easily load any 16 kHz audio and run inference using the same approach. In the next update we will use train/validate/test split on train-clean-100 data. And in final update we will use real world recording. 

If the audio is longer than 2 seconds, we can split it (like we do in training) into multiple 2s segments, denoise each, then concatenate results.

#### Code
Here is the [notebook](part3-first-update-files/audio_denoising_poc_1.ipynb) for part3. It has all the code and results discussed above.

**Notes on LLM Usage:** I used chatgpt for brainstorming and refining my ideas. And I used copilot autocomplete for fixing grammar and sentence structure in this document.