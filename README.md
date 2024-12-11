# BiPO: Bidirectional Partial Occlusion Network for Text-to-Motion Synthesis

Pytorch implementation of paper [BiPO: Bidirectional Partial Occlusion Network for Text-to-Motion Synthesis](https://arxiv.org/abs/2412.00112).

## Table of Content
- [1. Installation](#1-installation)
- [2. Train ParCo](#2-train-parco)
- [3. Evaluation](#3-evaluation)
- [Acknowledgement](#acknowledgement)

## 1. Installation

### 1.1. Environment

Our model was trained and tested on a single A6000-48G GPU
- Conda environment
  
  You need to following these scripts below to avoid potential package conflicts. 
  Otherwise, it may be unable to install PyTorch properly or fail to install some packages.

  - Create conda environment
    ```
    conda create -n BiPO blas=1.0 bzip2=1.0.8 ca-certificates=2021.7.5 certifi=2021.5.30 freetype=2.10.4 gmp=6.2.1 gnutls=3.6.15 intel-openmp=2021.3.0 jpeg=9b lame=3.100 lcms2=2.12 ld_impl_linux-64=2.35.1 libffi=3.3 libgcc-ng=9.3.0 libgomp=9.3.0 libiconv=1.15 libidn2=2.3.2 libpng=1.6.37 libstdcxx-ng=9.3.0 libtasn1=4.16.0 libtiff=4.2.0 libunistring=0.9.10 libuv=1.40.0 libwebp-base=1.2.0 lz4-c=1.9.3 mkl=2021.3.0 mkl-service=2.4.0 mkl_fft=1.3.0 mkl_random=1.2.2 ncurses=6.2 nettle=3.7.3 ninja=1.10.2 numpy=1.20.3 numpy-base=1.20.3 olefile=0.46 openh264=2.1.0 openjpeg=2.3.0 openssl=1.1.1k pillow=8.3.1 pip=21.0.1 readline=8.1 setuptools=52.0.0 six=1.16.0 sqlite=3.36.0 tk=8.6.10 typing_extensions=3.10.0.0 wheel=0.37.0 xz=5.2.5 zlib=1.2.11 zstd=1.4.9 python=3.7
    ```
    ```
    conda activate BiPO
    ```
  - Install essential packages (execute all scripts below)
    ```
    conda install ffmpeg=4.3 -c pytorch
    conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
    ```
    ``` 
    pip install absl-py==0.13.0 backcall==0.2.0 cachetools==4.2.2 charset-normalizer==2.0.4 chumpy==0.70 cycler==0.10.0 decorator==5.0.9 google-auth==1.35.0 google-auth-oauthlib==0.4.5 grpcio==1.39.0 idna==3.2 imageio==2.9.0 ipdb==0.13.9 ipython==7.26.0 ipython-genutils==0.2.0 jedi==0.18.0 joblib==1.0.1 kiwisolver==1.3.1 markdown==3.3.4 matplotlib==3.4.3 matplotlib-inline==0.1.2 oauthlib==3.1.1 pandas==1.3.2 parso==0.8.2 pexpect==4.8.0 pickleshare==0.7.5 prompt-toolkit==3.0.20 protobuf==3.17.3 ptyprocess==0.7.0 pyasn1==0.4.8 pyasn1-modules==0.2.8 pygments==2.10.0 pyparsing==2.4.7 python-dateutil==2.8.2 pytz==2021.1 pyyaml==5.4.1 requests==2.26.0 requests-oauthlib==1.3.0 rsa==4.7.2 scikit-learn==0.24.2 scipy==1.7.1 sklearn==0.0 smplx==0.1.28 tensorboard==2.6.0 tensorboard-data-server==0.6.1 tensorboard-plugin-wit==1.8.0 threadpoolctl==2.2.0 toml==0.10.2 tqdm==4.62.2 traitlets==5.0.5 urllib3==1.26.6 wcwidth==0.2.5 werkzeug==2.0.1 git+https://github.com/openai/CLIP.git git+https://github.com/nghorbani/human_body_prior gdown moviepy
    ```
    ```
    pip install imageio-ffmpeg
    pip install importlib-metadata==4.13.0
    ```

  - Install packages for rendering the motion (optional)
    ```
    bash dataset/prepare/download_smpl.sh
    conda install -c menpo osmesa
    conda install h5py
    conda install -c conda-forge shapely pyrender trimesh==3.22.5 mapbox_earcut
    ```

### 2.2. Feature extractors

We use the extractors provided by [T2M](https://github.com/EricGuo5513/text-to-motion) for evaluation.
Please download the extractors and glove word vectorizer. Note that 'zip' should be pre-installed in your system, if not, run
`sudo apt-get install zip` to install zip.
```
bash dataset/prepare/download_glove.sh
bash dataset/prepare/download_extractor.sh
```

### 2.3. Datasets

[HumanML3D](https://github.com/EricGuo5513/HumanML3D) is used by our project. 
You can find preparation and acquisition for the dataset [[here]](https://github.com/EricGuo5513/HumanML3D).

The file directory should look like:
```
./dataset/HumanML3D/
├── new_joint_vecs/
├── texts/
├── Mean.npy # same as in [HumanML3D](https://github.com/EricGuo5513/HumanML3D) 
├── Std.npy # same as in [HumanML3D](https://github.com/EricGuo5513/HumanML3D) 
├── train.txt
├── val.txt
├── test.txt
├── train_val.txt
└── all.txt
```

## 3. Train BiPO

The experiment directory structure of our project is:
```
./output  (arg.out_dir)
 ├── 00000-DATASET  (exp_number + dataset_name)
 │   └── VQVAE-EXP_NAME-DESC  (VQVAE + args.exp_name + desc)
 │       ├── events.out.XXX
 │       ├── net_best_XXX.pth
 │       ...
 │       ├── run.log
 │       ├── test_vqvae
 │       │   ├── ...
 │       │   ...
 │       ├── 0000-Trans-EXP_NAME-DESC  (stage2_exp_number + Trans + args.exp_name + desc)
 │       │   ├── quantized_dataset  (The quantized motion using VQVAE)
 │       │   ├── events.out.XXX
 │       │   ├── net_best_XXX.pth
 │       │   ...
 │       │   ├── run.log
 │       │   └── test_trans
 │       │       ├── ...
 │       │       ...
 │       ├── 0001-Trans-EXP_NAME-DESC
 │       ...
 ├── 00001-DATASET  (exp_number + dataset_name)
 ...
```

### 3.1. VQ-VAE
```bash
CUDA_VISIBLE_DEVICES=0 python train_BiPO_vq.py \
--out-dir output \
--exp-name BiPO \
--batch-size 256 \
--lr 2e-4 \
--total-iter 300000 \
--lr-scheduler 200000 \
--vqvae-cfg default \
--down-t 2 \
--depth 3 \
--dilation-growth-rate 3 \
--vq-act relu \
--quantizer ema_reset \
--loss-vel 0.5 \
--recons-loss l1_smooth
```

### 3.2. Text-to-Motion

Remember to set `--vqvae-train-dir` to the corresponding directory of the VQ-VAE you trained.

```bash
CUDA_VISIBLE_DEVICES=0 python train_BiPO_trans.py \
--vqvae-train-dir output/00000-t2m-BiPO/VQVAE-BiPO-t2m-default/ \
--select-vqvae-ckpt fid \
--exp-name ParCo \
--pkeep 0.4 \
--batch-size 64 \
--trans-cfg default \
--fuse-ver V1_3 \
--alpha 1.0 \
--num-layers 14 \
--embed-dim-gpt 1024 \
--nb-code 512 \
--n-head-gpt 16 \
--block-size 51 \
--ff-rate 4 \
--drop-out-rate 0.1 \
--total-iter 300000 \
--eval-iter 10000 \
--lr-scheduler 150000 \
--lr 0.0001 \
--dataname t2m \
--down-t 2 \
--depth 3 \
--quantizer ema_reset \
--dilation-growth-rate 3 \
--vq-act relu
```

## 4. Evaluation
### 4.1. VQ-VAE

Remember to set `--vqvae-train-dir` to the VQ-VAE you want to evaluate.
```bash
CUDA_VISIBLE_DEVICES=0 python eval_BiPO_vq.py --vqvae-train-dir output/00000-t2m-BiPO/VQVAE-BiPO-t2m-default/ --select-vqvae-ckpt fid
```

### 4.2. Text-to-Motion

If you want to evaluate the MultiModality (which takes a long time), just delete `--skip-mmod`.

Remember to set `--eval-exp-dir` to your trained BiPO's directory.
```bash
CUDA_VISIBLE_DEVICES=0 python eval_BiPO_trans_bi.py \
--eval-exp-dir output/00000-t2m-BiPO/VQVAE-BiPO-t2m-default/00000-Trans-BiPO-default \
--select-ckpt fid \
--skip-mmod
```

## Acknowledgement

We thank for:
- Public Codes: 
[ParCo](https://github.com/qrzou/ParCo),
[BAMM](https://github.com/exitudio/BAMM)
etc.
- Public Datase: [HumanML3D](https://github.com/EricGuo5513/HumanML3D).
