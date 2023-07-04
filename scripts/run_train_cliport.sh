set -e

# Train on 10
python train_transporter.py --tensorboard_dir=train_cliport_10 \
--checkpoint=gc_transporter_multitask_cliport_10.pt --batch_size=8 \
--constant_bg --epochs 150 --ndemos_train 10 --ndemos_test 50 \
--ndemos_val 10 \
--data_root benchmark_data \
--goal_conditioned_training --eval_freq 3 

# Eval on 10
python train_transporter.py \
--checkpoint=gc_transporter_multitask_cliport_10.pt --batch_size=8 \
--constant_bg --epochs 150 --ndemos_train 10 --ndemos_test 50 \
--ndemos_val 10 \
--data_root benchmark_data \
--goal_conditioned_training --eval_freq 3 --multi_task --eval_all \
--eval --eval_with_executor


# Train on 100
python train_transporter.py --tensorboard_dir=train_cliport_100 \
--checkpoint=gc_transporter_multitask_cliport_100.pt --batch_size=8 \
--constant_bg --epochs 150 --ndemos_train 100 --ndemos_test 50 \
--ndemos_val 10 \
--data_root benchmark_data \
--goal_conditioned_training --eval_freq 3 

# Eval on 100
python train_transporter.py \
--checkpoint=gc_transporter_multitask_cliport_100.pt --batch_size=8 \
--constant_bg --epochs 150 --ndemos_train 100 --ndemos_test 50 \
--ndemos_val 10 \
--data_root benchmark_data \
--goal_conditioned_training --eval_freq 3 --multi_task --eval_all \
--eval --eval_with_executor
