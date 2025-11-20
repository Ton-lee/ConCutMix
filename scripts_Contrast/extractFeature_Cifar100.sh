dataset="Cifar100"
data_root="/home/Users/dqy/Dataset/Cifar100/format_ImageNet/images/"
ckpt_path=""
save_root="/home/Users/dqy/Projects/ConCutMix/results/"
log_root="/home/Users/dqy/Projects/ConCutMix/log_Contrast"
extract_phase="val"
python /home/Users/dqy/Projects/ConCutMix/main.py\
  --data "${data_root}" \
  --lr 0.15 -p 194 --epochs 200 \
  --arch resnet32 \
  --wd 5e-4 \
  --cl_views uncutout-sim \
  --batch-size 256\
  --warmup_epochs 5\
  --feat_dim 128\
  --alpha 0 \
  --beta 1\
  --temp 0.1\
  --tau 1\
  --file_name baseline \
  --root_log "${log_root}"\
  --dataset "${dataset}"\
  --imb_factor 0.1\
  --l_d_warm 100\
  --topk 30\
  --scaling_factor 20 255\
  --num_classes 100 \
  --extract_feature \
  --extract_phase ${extract_phase} \
  --save_dir "${save_root}/${dataset}/features/${extract_phase}" \
  --reload True \
  --resume "${ckpt_path}"
