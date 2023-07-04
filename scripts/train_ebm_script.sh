python main_ebm.py --concept front --tensorboard_dir front_10 --checkpoint front_10.pt --use_kl --use_buffer --epochs 15 --n_samples 10

python main_ebm.py --concept behind --tensorboard_dir behind_10 --checkpoint behind_10.pt --use_kl --use_buffer --epochs 15 --n_samples 10

python main_ebm.py --concept left --tensorboard_dir left_10 --checkpoint left_10.pt --use_kl --use_buffer --epochs 15 --n_samples 10

python main_ebm.py --concept right --tensorboard_dir right_10 --checkpoint right_10.pt --use_kl --use_buffer --epochs 15 --n_samples 10

python main_ebm.py --concept inside --tensorboard_dir inside_10 --checkpoint inside_10.pt --use_kl --use_buffer --epochs 30 --n_samples 40

python main_ebm.py --concept circle --tensorboard_dir circle_10 --checkpoint circle_10.pt --use_kl --use_buffer --epochs 25 --n_samples 10

python main_ebm.py --concept line --tensorboard_dir line_10 --checkpoint line_10.pt --use_kl --use_buffer --epochs 25 --n_samples 10

python main_ebm.py --concept supported-by --tensorboard_dir supported_10 --checkpoint supported_10.pt --use_kl --use_buffer --epochs 5 --n_samples 10

python main_ebm.py --concept racer --tensorboard_dir racer_10 --checkpoint racer_10.pt --use_kl --use_buffer --epochs 25 --n_samples 10
