#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2,3
# Training
# bash tools/dist_train.sh configs/recognition/vit/vitclip_base_diving48.py 8 --test-last --validate \
# --cfg-options model.backbone.pretrained=openaiclip work_dir=work_dirs_vit/diving48/debug

# Evaluation only
# PORT=29667 bash tools/dist_test.sh configs/recognition/vit/flash_attn/vitclip_flash_base_diving48.py work_dirs/diving48/vitclip_diving48/baseline_flash_apex_gpunorm/best_top1_acc_epoch_38.pth 1 --eval top_k_accuracy --cfg-options fp16=True

# PORT=29667 bash tools/dist_test.sh configs/recognition/vit/AIM/AIM_flash_base_diving48.py \
#     work_dirs/diving48/vitclip_diving48/aim_flash_tcls_7x8-2x32_apex_gpunorm/epoch_50.pth 1 --eval top_k_accuracy --cfg-options fp16=True

# PORT=29667 bash tools/dist_test.sh configs/recognition/vit/vitclip_base_hmdb51.py work_dirs/hmdb51/vitclip_hmdb51/baseline_restuning_prompt_lamda_apex_gpunorm/best_top1_acc_epoch_1.pth 1 --eval top_k_accuracy


# PORT=29667 bash tools/dist_train.sh configs/recognition/vit/flash_attn/vitclip_flash_base_hmdb51.py 1 --test-best --validate --seed 0 --resume-from work_dirs/hmdb51/vitclip_hmdb51/tps_restuning_crsall_flash_apex_fd_gpunorm/latest.pth
# PORT=29667 bash tools/dist_train.sh configs/recognition/vit/flash_attn/vitclip_flash_base_diving48.py 1 --test-best --validate --seed 0

# PORT=29667 bash tools/dist_train.sh configs/recognition/vit/flash_attn/vitclip_flash_restuning_base_hmdb51.py 1 --test-best --validate --seed 0
# PORT=29667 bash tools/dist_train.sh configs/recognition/vit/flash_attn/vitclip_flash_restuning_base_diving48.py 1 --test-best --validate --seed 0 



# PORT=29667 bash tools/dist_train.sh configs/recognition/vit/zeroI2V/vitclip_zeroI2V_base_hmdb51.py 1 --test-best --validate --seed 0
# PORT=29667 bash tools/dist_test.sh configs/recognition/vit/zeroI2V/vitclip_zeroI2V_base_diving48.py work_dirs/diving48/vitclip_diving48/ths_tcls_ada_apex_acc/best_top1_acc_epoch_34.pth 1 --eval top_k_accuracy
# PORT=29667 bash tools/dist_train.sh configs/recognition/vit/zeroI2V/vitclip_zeroI2V_base_diving48.py 1 --test-best --validate --seed 0 --resume-from work_dirs/diving48/vitclip_diving48/ths_tcls_ada_apex_acc/best_top1_acc_epoch_34.pth


# PORT=29667 bash tools/dist_train.sh configs/recognition/vit/vitclip_base_hmdb51.py 1 --test-best --validate --seed 0
# PORT=29667 bash tools/dist_train.sh configs/recognition/vit/vitclip_base_diving48.py 1 --test-best --validate --seed 0


# bash tools/dist_train.sh configs/recognition/vit/AIM/AIM_flash_base_hmdb51.py 2 --validate --seed 0 
# PORT=29667 bash tools/dist_test.sh configs/recognition/vit/AIM/AIM_flash_base_hmdb51.py work_dirs/hmdb51/vitclip_hmdb51/aim_flash_tcls_2x2x32_shift_apex_gpunorm/best_top1_acc_epoch_*.pth 1 --eval top_k_accuracy --cfg-options fp16=True

# bash tools/dist_train.sh configs/recognition/vit/AIM/AIM_flash_base_diving48.py 2 --validate --seed 0 --resume-from work_dirs/diving48/vitclip_diving48/aim_flash_tcls_2x32_shift_apex_gpunorm/epoch_47.pth
# PORT=29667 bash tools/dist_test.sh configs/recognition/vit/AIM/AIM_flash_base_diving48.py work_dirs/diving48/vitclip_diving48/aim_flash_tcls_2x32_shift_apex_gpunorm/best_top1_acc_epoch_*.pth 1 --eval top_k_accuracy --cfg-options fp16=True


# PORT=29667 bash tools/dist_train.sh configs/recognition/vit/AIM/AIM_base_diving48.py 1 --test-best --validate --seed 0 
# PORT=29667 bash tools/dist_train.sh configs/recognition/vit/AIM/AIM_base_hmdb51.py 1 --validate --seed 0 --resume-from work_dirs/hmdb51/vitclip_hmdb51/aim_tcls_shift_2x32_apex_fd_gpunorm/epoch_22.pth



bash tools/dist_train.sh configs/recognition/vit/AIM/AIM_flash_win_base_hmdb51.py 2 --validate --seed 0 
# bash tools/dist_train.sh configs/recognition/vit/AIM/AIM_flash_win_base_diving48.py 2 --validate --seed 0 

# bash tools/dist_train.sh configs/recognition/vit/AIM/AIM_flash_win_base_sthv2.py 2 --test-last --validate --seed 0 

# bash tools/dist_train.sh configs/recognition/vit/AIM/AIM_flash_dual_base_hmdb51.py 2 --validate --seed 0 

# bash tools/dist_train.sh configs/recognition/vit/AIM/AIM_flash_base_ucf101.py 2 --validate --seed 0 
# bash tools/dist_train.sh configs/recognition/vit/AIM/AIM_flash_win_base_ucf101.py 2 --validate --seed 0 
# bash tools/dist_test.sh configs/recognition/vit/AIM/AIM_flash_base_ucf101.py work_dirs/ucf101/aim_flash_tcls_7x16_apex_gpunorm/best_top1_acc_epoch_13.pth 2 --eval top_k_accuracy --cfg-options fp16=True
