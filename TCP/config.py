import os

class GlobalConfig:
	seq_len = 1
	pred_len = 4

	root_dir_all = "bench2drive-base/"


	train_data = './tcp_bench2drive-train.npy'
	val_data = './tcp_bench2drive-val.npy'

	ignore_sides = True
	ignore_rear = True

	input_resolution = 256

	scale = 1
	crop = 256

	lr = 1e-4

	turn_KP = 0.75
	turn_KI = 0.75
	turn_KD = 0.3
	turn_n = 40

	speed_KP = 5.0
	speed_KI = 0.5
	speed_KD = 1.0
	speed_n = 40

	max_throttle = 0.75
	brake_speed = 0.4
	brake_ratio = 1.1
	clip_delta = 0.25


	aim_dist = 4.0
	angle_thresh = 0.3
	dist_thresh = 10


	speed_weight = 0.05
	value_weight = 0.001
	features_weight = 0.05

	rl_ckpt = "roach/log/ckpt_11833344.pth"

	img_aug = True


	def __init__(self, **kwargs):
		for k,v in kwargs.items():
			setattr(self, k, v)
