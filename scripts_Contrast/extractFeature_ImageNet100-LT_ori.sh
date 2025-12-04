dataset="ImageNet100-LT"
data_root="/home/Users/dqy/Dataset/ImageNet100-LT/format_ImageNet/images/"
ckpt_path="/home/Users/dqy/Projects/ConCutMix/log/baseline_ImageNet100-LT_resnet50_batchsize_256_epochs_100_temp_0.07_cutmix_prob_0.5_topk_30_scaling_factor_200_255_tau_0.99_lr_0.1_sim-sim/ConCutMix_ckpt.best.pth.tar"
save_root="/home/Users/dqy/Projects/ConCutMix/results_ori/"
log_root="/home/Users/dqy/Projects/ConCutMix/log"
extract_phase="train"
python /home/Users/dqy/Projects/ConCutMix/main.py\
  --data "${data_root}" \
  --lr 0.1 -p 300 --epochs 100 \
  --arch resnet50 \
  --use_norm \
  --wd 5e-4 \
  --cos \
  --cl_views sim-sim\
  --batch-size 256\
  --tau 0.99\
  --l_d_warm 70\
  --scaling_factor 200 255 \
  --topk 30\
  --num_classes 100\
  --imb_factor 1\
  --warmup_epochs 5\
  --alpha 0 \
  --beta 1\
  --file_name baseline \
  --root_log "${log_root}"\
  --dataset "${dataset}"\
  --extract_feature \
  --extract_phase ${extract_phase} \
  --save_dir "${save_root}/${dataset}/features/${extract_phase}" \
  --reload True \
  --resume "${ckpt_path}"
