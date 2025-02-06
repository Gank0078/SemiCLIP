torchrun --nproc_per_node 4 -m main --model "ViT-B-16" --pretrained openai --train-data "Fashion-ALL" --data-dir "Path/To/Dataset" \
--label-ratio "0.1" --val-data "Fashion200k" --imagenet-val "Fashion200k-CLS" --keyword-path "keywords/RS/class-name.txt" --lr 5e-5 --batch-size 64 \
--warmup 10 --zeroshot-frequency 5 --precision amp --method "semiclip" --seed "0" --stage 1 --epochs 25 --save_ckpt --pkname "stage1/fashion_semiclip_seed0" \
--pratio 0.3 --logs "../results/fashion_semiclip_stage1_seed0/"