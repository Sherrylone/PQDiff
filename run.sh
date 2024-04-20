# flickr 256
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_ldm.py --config=configs/flickr256_large.py
# flickr 192
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_ldm.py --config=configs/flickr192_large.py
# wikiart 192
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_ldm.py --config=configs/wikiart192_large.py
# building 192
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_ldm.py --config=configs/building192_large.py

# eval 1x
python3 -m torch.distributed.launch --nproc_per_node=8 \
        --node_rank 0 \
        --master_addr=${MASTER_ADDR:-127.0.0.1} \
        --master_port=${MASTER_PORT:-46123} \
        evaluate.py --target_expansion 0.25 0.25 0.25 0.25 --eval_dir ./eval_dir/scenery/1x/ --size 128 \
                --config flickr192_large
# eval 2x
python3 -m torch.distributed.launch --nproc_per_node=8 \
        --node_rank 0 \
        --master_addr=${MASTER_ADDR:-127.0.0.1} \
        --master_port=${MASTER_PORT:-36144} \
        evaluate.py --target_expansion 0.616 0.616 0.616 0.616 --eval_dir ./eval_dir/scenery/2x/ --size 86 \
                --config flickr192_large
# eval 3x
python3 -m torch.distributed.launch --nproc_per_node=8 \
        --node_rank 0 \
        --master_addr=${MASTER_ADDR:-127.0.0.1} \
        --master_port=${MASTER_PORT:-35123} \
        evaluate.py --target_expansion 1.214 1.214 1.214 1.214 --eval_dir ./eval_dir/scenery/3x/ --size 56 \
                --config flickr192_large

python eval_dir/inception.py --path ./eval_dir/scenery/1x/base_gen/generated/
python eval_dir/inception.py --path ./eval_dir/scenery/3x/copy/
python -m pytorch_fid ./eval_dir/scenery/1x/ori/ ./eval_dir/scenery/3x/copy/
python eval_dir/psnr.py --original ./eval_dir/scenery/2x/ori/ --contrast ./eval_dir/scenery/2x/gen/
python eval_dir/psnr.py --original ./eval_dir/building/1x/base_gen/gt/ --contrast ./eval_dir/building/1x/base_gen/generated/

# sample 
python3 -m torch.distributed.launch --nproc_per_node=8 \
        --node_rank 0 \
        --master_addr=${MASTER_ADDR:-127.0.0.1} \
        --master_port=${MASTER_PORT:-46123} samples.py \
                --target_expansion 0.25 0.25 0.25 0.25 \
                --eval_dir ./sample_dir/scenery/ \
                --size 128 \
                --config flickr192_large





