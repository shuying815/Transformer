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
We used a 24GB NVIDIA 3090 GPU for model training.

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

<img src="https://github.com/shuying815/Transformer/blob/main/results/超参配置表.png" width="420px">

<img src="https://github.com/shuying815/Transformer/blob/main/results/语言建模和机器翻译实验结果.png" width="420px">

<img src="https://github.com/shuying815/Transformer/blob/main/results/生成语言样例.png" width="800px">

<img src="https://github.com/shuying815/Transformer/blob/main/results/注意力头数量对比实验结果.png" width="420px">
<img src="https://github.com/shuying815/Transformer/blob/main/results/模型规模实验结果.png" width="420px">
<img src="https://github.com/shuying815/Transformer/blob/main/results/位置编码消融实验结果.png"width="420px">
<img src="https://github.com/shuying815/Transformer/blob/main/results/位置编码消融模型语言生成效果.png" width="900px">












