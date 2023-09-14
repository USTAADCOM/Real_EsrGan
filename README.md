# Real_EsrGan
Real-ESRGAN is an upgraded ESRGAN trained with pure synthetic data is capable of enhancing details while removing annoying artifacts for common real-world images.

## Setup
  ```code
  conda create -n <env_name>
  conda activate <env_name>
  git clone https://github.com/USTAADCOM/Real_EsrGan.git
  cd Real_EsrGan
  pip install -r requirements.txt -q
  ```
## Project Structure
```bash
Real_EsrGan
│   anime.png
│   app.py
│   inference_realesrgan.py
│   inference_realesrgan_video.py
│   README.md
│   requirements.txt
│   setup.cfg
│   setup.py
│   training.py
├───options
│       finetune_realesrgan_x4plus.yml
│       finetune_realesrgan_x4plus_pairdata.yml
│       setup.cfg
│       train_realesrgan_x2plus.yml
│       train_realesrgan_x4plus.yml
│       train_realesrnet_x2plus.yml
│       train_realesrnet_x4plus.yml
│
├───realesrgan
│   │   train.py
│   │   utils.py
│   │   __init__.py
│   │
│   ├───archs
│   │       discriminator_arch.py
│   │       srvgg_arch.py
│   │       __init__.py
│   │
│   ├───data
│   │       realesrgan_dataset.py
│   │       realesrgan_paired_dataset.py
│   │       __init__.py
│   │
│   ├───models
│   │       realesrgan_model.py
│   │       realesrnet_model.py
│   │       __init__.py
│   │
│   └───weights
├───scripts
│       extract_subimages.py
│       generate_meta_info.py
│       generate_meta_info_pairdata.py
│       generate_multiscale_DF2K.py
│       pytorch2onnx.py
│
└───tests
    │   test_dataset.py
    │   test_discriminator_arch.py
    │   test_model.py
    │   test_utils.py
    │
    └───data
        │   meta_info_gt.txt
        │   meta_info_pair.txt
        │   test_realesrgan_dataset.yml
        │   test_realesrgan_model.yml
        │   test_realesrgan_paired_dataset.yml
        │   test_realesrnet_model.yml
        │
        ├───gt
        │       baboon.png
        │       comic.png
        │
        ├───gt.lmdb
        │       data.mdb
        │       lock.mdb
        │       meta_info.txt
        │
        ├───lq
        │       baboon.png
        │       comic.png
        │
        └───lq.lmdb
                data.mdb
                lock.mdb
                meta_info.txt
```
## Run Gradio Demo
```code
python3 app.py 
```