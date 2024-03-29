export TASK_NAME=cola

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --fp16 \
  --save_total_limit 2 \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir $TASK_NAME/ \
  --overwrite_output_dir 
# rm -rf /$TASK_NAME/ws

# python run_glue.py --model_name_or_path bert-base-cased --task_name %TASK_NAME% --do_train --do_eval --fp16 --save_total_limit 2 --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 --output_dir %TASK_NAME%/ --overwrite_output_dir