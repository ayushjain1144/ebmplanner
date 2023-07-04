set -e


# Evaluation on 100 demos
python train_transporter.py --tensorboard_dir=relations_100 \
--checkpoint=gc_transporter_relations_100.pt --batch_size=8 \
--constant_bg --epochs 150 --ndemos_train 100 --ndemos_test 100 \
--ndemos_val 20 \
--data_root benchmark_dataer \
--goal_conditioned_training --eval_freq 10 \
--relations --theta_sigma 20 \
--multi_task --eval_all --eval --eval_with_executor

# Training on 100 demos
python train_transporter.py \
--checkpoint=gc_transporter_relations_100.pt --batch_size=8 \
--constant_bg --epochs 150 --ndemos_train 100 --ndemos_test 100 \
--ndemos_val 20 \
--data_root benchmark_dataer \
--goal_conditioned_training --eval_freq 1 \
--relations --theta_sigma 20


# Evaluation on 10 demos
python train_transporter.py --tensorboard_dir=relations_10 \
--checkpoint=gc_transporter_relations_10.pt --batch_size=8 \
--constant_bg --epochs 150 --ndemos_train 10 --ndemos_test 100 \
--ndemos_val 20 \
--data_root benchmark_dataer \
--goal_conditioned_training --eval_freq 10 \
--relations --theta_sigma 20 \
--multi_task --eval_all --eval --eval_with_executor

# Training on 10 demos
python train_transporter.py \
--checkpoint=gc_transporter_relations_10.pt --batch_size=8 \
--constant_bg --epochs 150 --ndemos_train 10 --ndemos_test 100 \
--ndemos_val 20 \
--data_root benchmark_dataer \
--goal_conditioned_training --eval_freq 10 \
--relations --theta_sigma 20







