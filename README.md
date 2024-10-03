# MOMA

Implementation for the paper "MoMA: Model-based Mirror Ascent for Offline Reinforcement Learning".

The dataset is provided in data/
To reproduce the data: 
python offline_gen.py --num_traj 50 --seed 121

To run the MOMA algorithm using the setup described in the paper:
python main.py --model_steps 150 --eta 0.1 --alpha 0.1 --mc_traj 300 --max_iter 40 --gamma 0.4
