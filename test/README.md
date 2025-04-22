Create a conda environment with the following command:
```bash
conda env create -f environment.yaml
```

Activate the environment:

```bash
conda activate audio_denoising_eval
```

To run the denoising script, use the following command:

```bash
python denoise_audio.py --model <model_path> --params <params_path> <input_file>

```
For example, to denoise the file `8555-292519-0015.flac` using the model `model_2s_10_epoch.pth` and parameters `model_2s_10_epoch_params.json`, run:

```
python denoise_audio.py --model model_2s_10_epoch.pth --params model_2s_10_epoch_params.json 8555-292519-0015.flac
```

