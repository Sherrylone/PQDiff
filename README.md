# Continuous-Multiple Image Outpainting in One-Step via Positional Query and A Diffusion-based Approach (ICLR 2024)
Authors: Shaofeng Zhang, Jinfa Huang, Qiang Zhou, Zhibin Wang, Fan Wang, Jiebo Luo, Junchi Yan

![](./framework.png)

This paper proposes PQDiff, which can outpaint images with continuous multiple and arbitrary positions.

**Dataset preparing**

We use Flickr, Buildings, and WikiArt datasets, which can be obtained at [link](https://github.com/Kaiseem/QueryOTR)

**Training stage**
```
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_ldm.py --config=configs/flickr192_large.py
```

You can train on your own dataset by modifying dataset/dataset.py

**Sampling stage**

We provide the 2.25x, 5x, and 11.7x outpainting settings (with copy operation). Run:

```
python3 -m torch.distributed.launch --nproc_per_node=8 \
        --node_rank 0 \
        --master_addr=${MASTER_ADDR:-127.0.0.1} \
        --master_port=${MASTER_PORT:-46123} \
        evaluate.py --target_expansion 0.25 0.25 0.25 0.25 --eval_dir ./eval_dir/scenery/1x/ --size 128 \
                --config flickr192_large
```

You can outpaint images with arbitrary and continuous multiples by changing the `target_expansion` parameters. The four parameters mean (top, down, left, right).

**Evaluation stage**

We provide scripts to evaluate inception scores, FID, and Centered PSNR scores in the `eval_dir`. Run:
```
python eval_dir/inception.py --path ./path1/
python -m pytorch_fid ./path1/ ./path2/
python eval_dir/psnr.py --original ./ori_dir/ --contrast ./gen_dir/
```

Here are some generated samples:
![](./samples.png)
