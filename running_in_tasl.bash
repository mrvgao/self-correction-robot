#!/bin/bash

tasl_xfmr_value_loss_lambda_5e-1_multi-task-with-value-
nohup /home/minquangao/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/tasl/lambda-0-point-1.json > tasl-logs/lambda-0-point-1.log &
nohup /home/minquangao/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/tasl/lambda-0-point-5.json > tasl-logs/lambda-0-point-5.log &
nohup /home/minquangao/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/tasl/lambda-1-point-5.json > tasl-logs/lambda-1-point-5.log &
nohup /home/minquangao/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/tasl/lambda_5.json > tasl-logs/lambda_5.log &
nohup /home/minquangao/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/tasl/lambda_50.json > tasl-logs/lambda_50.log &
nohup /home/minquangao/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/tasl/tasl_lambda_5_50_demos.json > tasl-logs/lambda_50_human_50.log &

nohup /home/ubuntu/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/LServer/lambda_5.json > tasl-logs/lambda_5.log &
nohup /home/ubuntu/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/LServer/lambda_5_50_demos.json > tasl-logs/lambda_5_50_demos.log &

nohup /home/ubuntu/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/LServer/rollout_lambda_5_1k_vloss_4e-2.json > tasl-logs/rollout_lambda_5_1k_vloss_4e.log &
nohup /home/ubuntu/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/LServer/rollout_lambda_5_1k_vloss_4e-2-running-2.json > tasl-logs/rollout_lambda_5_1k_vloss_4e_running_2.log &

nohup /home/ubuntu/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/LServer/rollout_lambda_5_1k_vloss_2e-2.json > tasl-logs/rollout_lambda_5_1k_vloss_2e.log &
nohup /home/ubuntu/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/LServer/rollout_lambda_5_1k_vloss_2e-2-running-2.json > tasl-logs/rollout_lambda_5_1k_vloss_2e_running_2.log &

nohup /home/ubuntu/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/LServer/rollout_lambda_5_450_vloss_2e-2.json > tasl-logs/rollout_lambda_5_epoch_450_vloss_2e.log &
nohup /home/ubuntu/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/LServer/rollout_lambda_5_450_vloss_2e-2-running-2.json > tasl-logs/rollout_lambda_5_epoch_450_vloss_2e_running_2.log &

nohup /home/ubuntu/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/LServer/rollout_lambda_5_450_vloss_4e-2.json > tasl-logs/rollout_lambda_5_epoch_450_vloss_4e.log &
nohup /home/ubuntu/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/LServer/rollout_lambda_5_450_vloss_4e-2-running-2.json > tasl-logs/rollout_lambda_5_epoch_450_vloss_4e_running_2.log &

nohup /home/ubuntu/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/LServer/rollout_lambda_5_1k_vloss_5e-2-seed-0.json > tasl-logs/rollout_lambda_5_1k_vloss_5e-2-seed-0.log &
nohup /home/ubuntu/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/LServer/rollout_lambda_5_1k_vloss_5e-2-seed-100.json > tasl-logs/rollout_lambda_5_1k_vloss_5e-2-seed-100.log &
nohup /home/ubuntu/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/LServer/rollout_lambda_5_1k_vloss_5e-2-seed-200.json > tasl-logs/rollout_lambda_5_1k_vloss_5e-2-seed-200.log &
nohup /home/ubuntu/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/LServer/rollout_lambda_5_1k_vloss_5e-2-seed-500.json > tasl-logs/rollout_lambda_5_1k_vloss_5e-2-seed-500.log &
nohup /home/ubuntu/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/LServer/rollout_lambda_5_1k_vloss_5e-2-seed-700.json > tasl-logs/rollout_lambda_5_1k_vloss_5e-2-seed-700.log &
nohup /home/ubuntu/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/LServer/rollout_lambda_5_1k_vloss_5e-2-seed-999.json > tasl-logs/rollout_lambda_5_1k_vloss_5e-2-seed-999.log &

nohup /home/minquangao/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/tasl/rollout/rollout_lambda_5_1k_vloss_5e-2-seed-999.json > tasl-logs/rollout_lambda_5_1k_vloss_5e-2-seed-999-2.log &
# running tail 999-2.log
nohup /home/minquangao/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/tasl/rollout/rollout_lambda_5_1k_vloss_5e-2-seed-100.json > tasl-logs/rollout_lambda_5_1k_vloss_5e-2-seed-100-2.log &
# running tail 100-2.log
nohup /home/minquangao/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/tasl/rollout/rollout_lambda_5_1k_vloss_5e-2-seed-200.json > tasl-logs/rollout_lambda_5_1k_vloss_5e-2-seed-200.log &
nohup /home/minquangao/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/tasl/rollout/rollout_lambda_5_1k_vloss_5e-2-seed-500.json > tasl-logs/rollout_lambda_5_1k_vloss_5e-2-seed-500-2.log &
# running tail 500.log
nohup /home/minquangao/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/tasl/rollout/rollout_lambda_5_1k_vloss_5e-2-seed-700.json > tasl-logs/rollout_lambda_5_1k_vloss_5e-2-seed-700.log &
nohup /home/minquangao/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/tasl/rollout/rollout_lambda_5_1k_vloss_5e-2-seed-0.json > tasl-logs/rollout_lambda_5_1k_vloss_5e-2-seed-0.log &

wait