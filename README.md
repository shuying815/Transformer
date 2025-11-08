# Transformer

### Pipeline

<img src="https://github.com/shuying815/Transformer/blob/main/fig/Transformer_full_architecture.png" style="zoom:50%;" />

### Installation

```
conda create -n transformer python=3.8
conda activate transformer
pip intsall -r requirements.txt
```

### Prepare Dataset

Download the datasets: [WikiText-2](https://huggingface.co/datasets/mindchain/wikitext2)  for encoder-only transformer training,  [IWSLT2017](https://huggingface.co/datasets/mindchain/wikitext2)  for encoder-decoder transformer training.  And then unzip them to `your_dataset_dir`.

### Training

If you want to do encoder_only transformer training, you can run (or run train.py):

``` 
bash scripts/run.sh
```

If you want to do encoder_decoder transformer training, you can run:

```
python train.py --config_file ./configs/iwslt.yml
```

If you want to modify transformer architecture, you need to modify the model configuration in`./configs/XXX.yml` or input from the command line. Default seed value is 2.

### Evaluation

For example, if you want to test transformer for machine translation mask:

```
python test.py --config_file ./configs/iwslt.yml
```

### Training Results
All results are stored in `results' directory.






