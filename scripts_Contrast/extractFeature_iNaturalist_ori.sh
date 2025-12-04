dataset="iNaturalist"
data_root="/home/Users/dqy/Dataset/iNaturalist/"
ckpt_path="/home/Users/dqy/Projects/ConCutMix/log/baseline_iNaturalist_resnet50_batchsize_128_epochs_100_temp_0.07_cutmix_prob_0.5_topk_30_scaling_factor_1628_255_tau_0.99_lr_0.2_sim-sim/ConCutMix_ckpt.best_scl.pth.tar"
save_root="/home/Users/dqy/Projects/ConCutMix/results_ori/"
log_root="/home/Users/dqy/Projects/ConCutMix/log"
extract_phase="val"
python /home/Users/dqy/Projects/ConCutMix/main.py\
  --data "${data_root}" \
  --lr 0.2 -p 600 --epochs 100 \
  --arch resnet50 \
  --use_norm  \
  --wd 1e-4 \
  --cos \
  --cl_views sim-sim\
  --batch-size 128\
  --tau 0.99\
  --l_d_warm 80\
  --scaling_factor 1628 255 \
  --grad_c\
  --topk 30\
  --num_classes 8142\
  --file_name baseline \
  --root_log "${log_root}"\
  --dataset "${dataset}"\
  --extract_feature \
  --extract_phase ${extract_phase} \
  --save_dir "${save_root}/${dataset}/features/${extract_phase}" \
  --reload True \
  --resume "${ckpt_path}"
