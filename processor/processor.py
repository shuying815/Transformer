from src.model.transformer import causal_mask
from processor.wikiprocessor import compute_token_accuracy, visualize
import time
import torch.nn as nn
import torch
import os
import numpy as np

def train(epochs, cfg, model, n_tokens, optimizer, loss, dataloader, device = "cuda", scheduler = None):
    
    model.to(device)
    # 从上一次训练截断处继续
    if cfg.RESUME_CKPT != '':
        model.load_state_dict(cfg.RESUME_CKPT)

    train_loss_list = []
    train_acc_list = []
    iter_loss_list = []
    iter_acc_list = []

    start_time = time.time()

    # 训练
    for epoch in range(epochs):

        model.train()
        scheduler.step()
        epoch_loss = 0
        epoch_acc = 0
        n = 0
        n_iter = 0

        for data in dataloader:
            t_src, t_tgt = data['input_ids'], data['labels']
            src_mask, tgt_mask = data['src_attn_mask'], data['tgt_attn_mask'] # padding mask
            B = t_tgt.shape[0]
        
            t_src = t_src.to(device)
            t_tgt = t_tgt.to(device)
            src_mask = src_mask.to(device)
            src_mask = src_mask.unsqueeze(1).unsqueeze(2) # 升维，方便多头注意力操作
            tgt_mask = tgt_mask.to(device)
            padding_mask = tgt_mask.unsqueeze(1)

            # 目标语言需要结合padding mask和causal mask
            tgt_total_mask = padding_mask & causal_mask(t_tgt).to(device)
            tgt_total_mask = tgt_total_mask.unsqueeze(1)

            output = model(t_src, t_tgt, src_mask = src_mask, tgt_mask = tgt_total_mask)
            output = output.reshape(-1, n_tokens)
            t_tgt = t_tgt.reshape(-1)
            l = loss(output, t_tgt)

            optimizer.zero_grad()
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            
            acc = compute_token_accuracy(output.reshape(B, -1, n_tokens), t_tgt.reshape(B, -1), tgt_mask)
            epoch_acc += acc * B
            n += B
            n_iter += 1
            epoch_loss += l.item()
            print("iter{}, iter train loss:{:.3f}, iter train accuracy:{:.1%}".format(n_iter, l.item(), acc))

            iter_loss_list.append(l.item())
            iter_acc_list.append(acc)

        train_loss_list.append(epoch_loss / (n_iter + 1))
        train_acc_list.append(epoch_acc / n)

        if epoch % cfg.TRAIN.SAVECKPT_PERIOD == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, f'{epoch + 1}_ckpt.pth'))

        print("****epoch{}, train loss:{:.3f}, train accuracy:{:.1%}".format(epoch + 1, train_loss_list[-1], train_acc_list[-1]))

    end_time = time.time()

    # 统计模型参数量
    total_params = sum(p.numel() for p in model.parameters())

    # 保存所有结果
    metric_dict = {}
    metric_dict['train_loss_list'] = train_loss_list
    metric_dict['train_acc_list'] = train_acc_list
    metric_dict['iter_loss_list'] = iter_loss_list
    metric_dict['iter_acc_list'] = iter_acc_list
    metric_dict['train_time'] = end_time - start_time
    metric_dict['total_params'] = total_params
    np.save(os.path.join(cfg.OUTPUT_DIR, 'metric.npy'), metric_dict)

    # 可视化训练曲线
    visualize(cfg.OUTPUT_DIR, train_loss_list, train_acc_list)
    visualize(cfg.OUTPUT_DIR, iter_loss_list, iter_acc_list, t = 'iter')
    
    return train_loss_list, train_acc_list, end_time - start_time