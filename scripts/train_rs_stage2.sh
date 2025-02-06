torchrun --nproc_per_node 4 -m main --model "ViT-B-16" --pretrained openai --train-data "RS-ALL" --data-dir "Path/To/Dataset" \
--label-ratio "0.1" --val-data "RS-ALL" --imagenet-val "RSICD-CLS" --keyword-path "keywords/RS/class-name.txt" --lr 5e-5 --batch-size 64 \
--warmup 10 --zeroshot-frequency 5 --precision amp --method "semiclip" --seed "0" --stage 2 --epochs 15 --pkname "stage1/rs_semiclip_seed0" --pratio 0.3 --save_ckpt --logs "../results/rs_semiclip_stage2_seed0/" \
--resume-path "../results/rs_semiclip_stage1_seed0/checkpoints"