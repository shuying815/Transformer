import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from src.model.transformer import causal_mask

# -----------------------------------------------------------------------------
# 训练
# 指标计算、结果记录、曲线可视化
# -----------------------------------------------------------------------------

# 计算token预测准确率
# logits: 模型预测值
# labels: 标签值
def compute_token_accuracy(logits, labels, mask = None):
    """
    logits: [batch_size, seq_len, vocab_size]
    labels: [batch_size, seq_len]
    """
    # 取预测值
    preds = logits.argmax(dim=-1)  # [batch_size, seq_len]
    # 忽略 padding token（如果存在）
    if mask == None:
        mask = labels != -100  # -100是CrossEntropyLoss的ignore_index
    correct = (preds == labels) & mask
    # 计算准确率
    acc = correct.sum().float() / mask.sum().float()
    return acc.item()

# 可视化loss、acc曲线，结果保存为pdf
# path: 保存路径
# loss_list: 损失列表
# acc_list: 准确率列表
def visualize(path, loss_list, acc_list, t = 'train'):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(loss_list, label='Loss')
    plt.xlabel('epoch')
    plt.ylabel('train loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(acc_list, label='accuracy')
    plt.xlabel('epoch')
    plt.ylabel('train accuracy')
    plt.legend()

    plt.savefig(os.path.join(path, f'curve_{t}.pdf'))
    plt.close()

# 训练并保存模型，并在测试集上测试
def wikitrain(epochs, cfg, model, wikitext, optimizer, loss, train_data, test_data, device = "cuda", scheduler = None):

    model.to(device)
    # 从上一次训练截断处继续
    if cfg.RESUME_CKPT != '':
        model.load_state_dict(cfg.RESUME_CKPT)

    train_loss_list = []
    train_acc_list = []
    iter_loss_list = []
    iter_acc_list = []

    bptt = cfg.DATASET.BPTT
    n_tokens = len(wikitext.tokenizer)  # 不重复词汇

    start_time = time.time()

    # 训练
    for epoch in range(epochs):

        model.train()
        scheduler.step()
        epoch_loss = 0
        epoch_acc = 0
        n = 0

        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            t_src, t_tgt = wikitext.get_batch(train_data, i)
            t_src.to(device)
            t_tgt.to(device)

            t_src = t_src.t()
            t_tgt = t_tgt.t()
            tgt_mask = casual_mask(t_tgt)

            output = model(t_src, t_tgt, tgt_mask = tgt_mask)
            output = output.reshape(-1, n_tokens)
            t_tgt = t_tgt.reshape(-1)
            l = loss(output, t_tgt)

            optimizer.zero_grad()
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            acc = compute_token_accuracy(output, t_tgt)
            epoch_acc += acc * t_tgt.shape[0]
            n += t_tgt.shape[0]
            epoch_loss += l.item()
            print("iter{}, iter train loss:{:.3f}, iter train accuracy:{:.1%}".format(batch + 1, l.item(), acc))

            iter_loss_list.append(l.item())
            iter_acc_list.append(acc)

        train_loss_list.append(epoch_loss / (batch + 1))
        train_acc_list.append(epoch_acc / n)

        if epoch % cfg.TRAIN.SAVECKPT_PERIOD == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, f'{epoch + 1}_ckpt.pth'))

        print("****epoch{}, train loss:{:.3f}, train accuracy:{:.1%}".format(epoch + 1, train_loss_list[-1], train_acc_list[-1]))

    end_time = time.time()

    #测试
    with torch.no_grad():
        model.eval()
        test_loss = 0
        test_acc = 0
        n = 0
        for batch, i in enumerate(range(0, test_data.size(0) - 1, bptt)):
            t_src, t_tgt = wikitext.get_batch(test_data, i)
            t_src.to(device)
            t_tgt.to(device)

            t_src = t_src.t()
            t_tgt = t_tgt.t()

            tgt_mask =  casual_mask(t_tgt)
            output = model(t_src, t_tgt, tgt_mask = tgt_mask)
            output = output.reshape(-1, n_tokens)
            t_tgt = t_tgt.reshape(-1)
            l = loss(output, t_tgt)

            acc = compute_token_accuracy(output, t_tgt)
            test_acc += acc * t_tgt.shape[0]
            n += t_tgt.shape[0]
            test_loss += l.item()

        test_acc /= n
        test_loss /= (batch + 1)

        print("test loss:{:.3f}, test accuracy:{:.1%}".format(test_loss, test_acc))

    # 统计模型参数量
    total_params = sum(p.numel() for p in model.parameters())

    # 保存所有结果
    metric_dict = {}
    metric_dict['train_loss_list'] = train_loss_list
    metric_dict['train_acc_list'] = train_acc_list
    metric_dict['iter_loss_list'] = iter_loss_list
    metric_dict['iter_acc_list'] = iter_acc_list
    metric_dict['test_loss'] = test_loss
    metric_dict['test_acc'] = test_acc
    metric_dict['train_time'] = end_time - start_time
    metric_dict['total_params'] = total_params
    np.save(os.path.join(cfg.OUTPUT_DIR, 'metric.npy'), metric_dict)

    # 可视化训练曲线
    visualize(cfg.OUTPUT_DIR, iter_loss_list, iter_acc_list)

    return train_loss_list, train_acc_list, test_loss, test_acc, end_time - start_time