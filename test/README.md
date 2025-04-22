
This folder contains `denoise_audio.py` script that can be used to denoise audio files using the trained model. The `models` folder contains the trained models and parameters. The required dependencies are listed in the `environment.yaml` file.

To get started, create a conda environment using the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
```

Then activate the environment:
```bash
conda activate audio_denoising_eval
```

To run the denoising script, use the following command:

```bash
python denoise_audio.py -i <path_to_noisy_audio> -o <output_path> -m <model_path> -p <params_path>
```

For example, to denoise a sample audio file in the `noisy_sample` folder, run the following command:

```bash
python denoise_audio.py \
-i noisy_sample/noisy-sample-1-syn-noise.wav \
-o denoise_output.wav \
-m models/best_finetuned_model.pth \
-p models/best_finetuned_model_params.json             
```
To get all the available options, run the following command:

```bash
python denoise_audio.py -h
```

Here is the output of the command:

```
usage: denoise_audio.py [-h] -i NOISY_AUDIO [-o OUTPUT] [-m MODEL] [-p PARAMS] [--overlap OVERLAP]

Denoise long audio files with DesnoingUNet Autoencoder

options:
  -h, --help            show this help message and exit
  -i NOISY_AUDIO, --noisy-audio NOISY_AUDIO
                        Path to noisy input audio (.flac/.wav)
  -o OUTPUT, --output OUTPUT
                        Output path
  -m MODEL, --model MODEL
                        Model file path
  -p PARAMS, --params PARAMS
                        Params file path
  --overlap OVERLAP     Overlap ratio between segments
  ````