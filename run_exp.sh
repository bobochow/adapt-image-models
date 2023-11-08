#!/usr/bin/env bash

# Training
# bash tools/dist_train.sh configs/recognition/vit/vitclip_base_diving48.py 8 --test-last --validate \
# --cfg-options model.backbone.pretrained=openaiclip work_dir=work_dirs_vit/diving48/debug

# Evaluation only
# PORT=29667 bash tools/dist_test.sh configs/recognition/vit/flash_attn/vitclip_flash_base_diving48.py work_dirs/diving48/vitclip_diving48/baseline_flash_apex_imgaug/best_top1_acc_epoch_42.pth 1 --eval top_k_accuracy

PORT=29667 bash tools/dist_train.sh configs/recognition/vit/flash_attn/vitclip_flash_base_hmdb51.py 1 --test-best --validate --seed 0
# PORT=29667 bash tools/dist_train.sh configs/recognition/vit/flash_attn/vitclip_flash_base_diving48.py 1 --test-best --validate --seed 0
