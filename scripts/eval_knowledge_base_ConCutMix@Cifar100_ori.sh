#!/bin/bash
# 以先验权重 0.0 测试原始模型性能

DATASET="Cifar100"
checkpoint="/home/Users/dqy/Projects/ConCutMix/log/baseline_Cifar100_resnet32_batchsize_256_epochs_200_temp_0.1_cutmix_prob_0.5_topk_30_scaling_factor_20_255_tau_1.0_lr_0.15_uncutout-sim/ConCutMix_ckpt.best_acc1.pth.tar"
save_root="/home/Users/dqy/Projects/ConCutMix/results_${DATASET}/distorted_with_knowledge_ConCutMix"
script_path="/home/Users/dqy/Projects/ConCutMix/main_ConCutMix@${DATASET}.py"
dataset_root="/home/Users/dqy/Dataset/${DATASET}/format_ImageNet/"
knowledge_root="${dataset_root}/KnowledgeBase_train_ConCutMix@ConCutMix"
log_root="/home/Users/dqy/Projects/ConCutMix/log"

trap 'cleanup' INT

cleanup() {
    echo -e "\n[!] 用户中断 (Ctrl+C)。终止所有子进程..."
    # 杀死所有相关后台进程
    pkill -P $$  # 杀死当前脚本的所有子进程
    pkill -f "main_Contrast@${DATASET}"   # 确保 torchrun 进程被杀死（可选）
    exit 1       # 退出脚本
    echo -e "\n[!] 用户中断 (Ctrl+C)。已终止所有子进程"
}

# 测试原始数据的性能结果
distortion_type="none"
distortion_param=0
output_file="${save_root}/performance_${distortion_type}.csv"
mkdir -p "${save_root}"
# echo "distortion_type,distortion_param,prior_weight,knowledge_type,knowledge_K,retrieval_k,performance" > "${output_file}"
device="4,5"

# 进度统计变量
start_time=$(date +%s)
total_iterations=$((5 * 1 * 5 * 5))  # 4*2*5*5=200种组合
current_iteration=0

# 计算剩余时间函数
calculate_remaining_time() {
    local elapsed=$1
    local completed=$2
    local total=$3
    if [ $completed -eq 0 ]; then
        echo "N/A"
    else
        local remaining=$(( (total - completed) * elapsed / completed ))
        printf "%02d:%02d:%02d" $((remaining/3600)) $(( (remaining%3600)/60 )) $((remaining%60))
    fi
}

  distortion_name="${distortion_type}"
  for prior_weight in "0.0"; do
      for knowledge_type in "GMM_category"; do
          for knowledge_K in 100; do
              knowledge_name="${knowledge_type}@K=${knowledge_K}"
              knowledge_path="${knowledge_root}/${knowledge_name}/"
              for retrieval_k in 1; do
                  current_iteration=$((current_iteration + 1))
                  current_time=$(date +%s)
                  elapsed=$((current_time - start_time))

                  # 计算进度百分比
                  progress=$((100 * current_iteration / total_iterations))

                  # 计算剩余时间
                  remaining=$(calculate_remaining_time $elapsed $current_iteration $total_iterations)

                  # 显示进度信息
                  echo -ne "\r[${progress}%] Iteration ${current_iteration}/${total_iterations} | "
                  echo -ne "Elapsed: $(printf "%02d:%02d:%02d" $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))) | "
                  echo -ne "Remaining: ${remaining} | "
                  echo -ne "Current params: ${distortion_type}@${distortion_param}, w=${prior_weight}, ${knowledge_type}@K=${knowledge_K}, top-${retrieval_k}"
                  echo ""
                  # 运行模型
                  CUDA_VISIBLE_DEVICES=${device} /home/Users/dqy/miniconda3/envs/SKB/bin/python "${script_path}" \
                    --data "${dataset_root}/images/" \
                    --lr 0.15 -p 194 --epochs 200 \
                    --arch resnet32 \
                    --wd 5e-4 \
                    --cl_views uncutout-sim \
                    --batch-size 256\
                    --feat_dim 128\
                    --tau 1\
                    --temp 0.1\
                    --l_d_warm 100\
                    --scaling_factor 20 255 \
                    --topk 30\
                    --num_classes 100\
                    --imb_factor 1\
                    --warmup_epochs 5\
                    --alpha 2 \
                    --beta 0.6\
                    --file_name baseline \
                    --root_log "${log_root}"\
                    --dataset "${DATASET}"\
                    --save_dir "${save_root}/weight=${prior_weight}/${knowledge_name}/top-${retrieval_k}/${distortion_name}/" \
                    --reload True \
                    --resume "${checkpoint}" \
                    --knowledge_base "${knowledge_path}" \
                    --prior_weight ${prior_weight} \
                    --retrieval_k ${retrieval_k} \
                    --result_json_path "${save_root}/weight=${prior_weight}/${knowledge_name}/top-${retrieval_k}/${distortion_name}/result.json" \
                    --device_ids 0 1

                  # # 从日志文件中提取性能指标
                  # log_file="${save_root}/weight=${prior_weight}/${knowledge_name}/top-${retrieval_k}/${distortion_name}/log.txt"
                  # if [ -f "${log_file}" ]; then
                  #     # 获取第4行，然后取最后一个空格后的内容
                  #     performance=$(sed -n '5p' "${log_file}" | awk '{print $NF}')

                  #     # 将结果写入CSV文件
                  #     echo "${distortion_type},${distortion_param},${prior_weight},${knowledge_type},${knowledge_K},${retrieval_k},${performance}" >> "${output_file}"
                  # else
                  #     echo "Log file not found: ${log_file}"
                  #     echo "${distortion_type},${distortion_param},${prior_weight},${knowledge_type},${knowledge_K},${retrieval_k},NA" >> "${output_file}"
                  # fi
              done
          done
      done
  done
echo -e "\nAll tests completed. Results saved to ${output_file}"