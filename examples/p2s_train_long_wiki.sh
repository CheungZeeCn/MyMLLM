OUTPUT_DIR=/home/ana/data4/output_models/MyMLLM/p2s_pretrain/stage0_longwiki_80w
TEE_FILE="${OUTPUT_DIR}/tee_out.txt"
mkdir -p $OUTPUT_DIR

echo "OUTPUT: ${OUTPUT_DIR}"
echo "teefile: ${TEE_FILE}"

python base_pretrain.py   \
  --predict_with_generate   --max_steps 150000  --gradient_accumulation_steps 64  \
   --per_device_train_batch_size 1 --per_device_eval_batch_size 1    \
  --evaluation_strategy steps  --save_steps 3000 --eval_steps 6000 --logging_steps 10   \
   --overwrite_output_dir  \
  --output_dir $OUTPUT_DIR  \
   --data_dir=/home/ana/data4/datasets/wiki_zh/synthdog_out/cn_wiki_500k_long \
   --lr_scheduler_type cosine   --warmup_steps 1000   --optim adafactor   --learning_rate 1e-4 \
   --generation_max_length 2048   --generation_num_beams 1  --gradient_checkpointing \
   --fp16  2>&1 | tee -a $TEE_FILE

