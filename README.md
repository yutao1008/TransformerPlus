# Horizontal and Vertical Attention in Transformers: A Pytorch Implementation

This is a PyTorch implementation of the Transformer model in "[Horizontal and Vertical Attention in Transformers](https://arxiv.org/pdf/2207.04399.pdf)" (Litao Yu and Jian Zhang). The codes are modified from the public Pytorch implementation at (https://github.com/jadore801120/attention-is-all-you-need-pytorch).



The project support training and translation with trained model now.



If there is any suggestion or error, feel free to fire an issue to let me know. :)


# Usage

## WMT'16 Multimodal Translation: de-en

An example of training for the WMT'16 Multimodal Translation task (http://www.statmt.org/wmt16/multimodal-task.html).

### 0) Download the spacy language model.
```bash
# conda install -c conda-forge spacy 
python -m spacy download en
python -m spacy download de
```

### 1) Preprocess the data with torchtext and spacy.
```bash
python preprocess.py -lang_src de -lang_trg en -share_vocab -save_data m30k_deen_shr.pkl
```

### 2) Train the model
```bash
python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -output_dir output -b 256 -warmup 128000 -epoch 400
```

### 3) Test the model
```bash
python translate.py -data_pkl m30k_deen_shr.pkl -model trained.chkpt -output prediction.txt
```

## [(WIP)] WMT'17 Multimodal Translation: de-en w/ BPE 
### 1) Download and preprocess the data with bpe:

> Since the interfaces is not unified, you need to switch the main function call from `main_wo_bpe` to `main`.

```bash
python preprocess.py -raw_dir /tmp/raw_deen -data_dir ./bpe_deen -save_data bpe_vocab.pkl -codes codes.txt -prefix deen
```

### 2) Train the model
```bash
python train.py \
  -data_pkl m30k_deen_shr.pkl \
  -label_smoothing \
  -proj_share_weight \
  -scale_emb_or_prj emb \
  -h_attn \
  -v_attn \
  -lr_mul 0.5 \
  -b 512 \
  -warmup 4000 \
  -epoch 100 \
  -output_dir output/wmt16_hv 
```

### 3) Test the model (not ready)
- TODO:
	- Load vocabulary.
	- Perform decoding after the translation.
---
# Performance
## Training

<p align="center">
<img src="https://i.imgur.com/S2EVtJx.png" width="400">
<img src="https://i.imgur.com/IZQmUKO.png" width="400">
</p>

- Parameter settings:
  - batch size 512 
  - warmup step 4000 
  - epoch 200 
  - lr_mul 0.5
  - label smoothing 
  - do not apply BPE and shared vocabulary
  - target embedding / pre-softmax linear layer weight sharing. 
 
  
## Testing 
- coming soon.
---
# TODO
  - Evaluation on the generated text.
  - Attention weight plot.
---
# Acknowledgement
- The byte pair encoding parts are borrowed from [subword-nmt](https://github.com/rsennrich/subword-nmt/).
- The project structure, some scripts and the dataset preprocessing steps are heavily borrowed from [OpenNMT/OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).
- Thanks for the suggestions from @srush, @iamalbert, @Zessay, @JulesGM, @ZiJianZhao, and @huanghoujing.
