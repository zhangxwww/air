运行方法：

首先将cifar100数据集放在dataset目录下

```
python -u baseline_v4.py --mse --bce --model_name b4_r_2_5_ema1_normbefore_save --ssl_lr 1e-2 --ssl_epochs 100 --ce_lr 5e-2 --cluster_lr 5e-2 --cluster_epochs 100 --first_ssl --first_ce --herding --geo --geo_dist cosine --geo_k 15 --geo_percent 0.5 --thres1_ratio 2.0 --thres2_ratio 5.0 --ema_beta 1.0 --norm_before_add --print_cls_statistics --save_exemplars
```