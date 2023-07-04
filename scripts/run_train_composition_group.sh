set -e

# Note: No training needed

# Evaluating on 10
python train_transporter.py \
--checkpoint=gc_transporter_relations_10.pt --batch_size=3 \
--constant_bg --epochs 150 --ndemos_train 10 --ndemos_test 100 \
--ndemos_val 10 \
--data_root benchmark_data \
--goal_conditioned_training \
--eval_all --eval_freq 1 --multi_relations_group --theta_sigma 20 \
--eval --eval_with_executor  


# Evaluating on 100
python train_transporter.py \
--checkpoint=gc_transporter_relations_100.pt --batch_size=3 \
--constant_bg --epochs 150 --ndemos_train 100 --ndemos_test 100 \
--ndemos_val 10 \
--data_root benchmark_data \
--goal_conditioned_training \
--eval_all --eval_freq 1 --multi_relations_group --theta_sigma 20 \
--eval --eval_with_executor