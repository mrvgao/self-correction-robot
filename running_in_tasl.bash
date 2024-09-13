#!/bin/bash

tasl_xfmr_value_loss_lambda_5e-1_multi-task-with-value-
nohup /home/minquangao/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/tasl/lambda-0-point-1.json > tasl-logs/lambda-0-point-1.log &
nohup /home/minquangao/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/tasl/lambda-0-point-5.json > tasl-logs/lambda-0-point-5.log &
nohup /home/minquangao/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/tasl/lambda-1-point-5.json > tasl-logs/lambda-1-point-5.log &
nohup /home/minquangao/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/tasl/lambda_5.json > tasl-logs/lambda_5.log &
nohup /home/minquangao/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/tasl/lambda_50.json > tasl-logs/lambda_50.log &
nohup /home/ubuntu/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/LServer/lambda_5.json > tasl-logs/lambda_5.log &
nohup /home/ubuntu/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/LServer/lambda_5_50_demos.json > tasl-logs/lambda_5_50_demos.log &

nohup /home/ubuntu/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/LServer/rollout_lambda_5_1k_vloss_4e-2.json > tasl-logs/rollout_lambda_5_1k_vloss_4e.log &
nohup /home/ubuntu/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/LServer/rollout_lambda_5_1k_vloss_2e-2.json > tasl-logs/rollout_lambda_5_600_vloss_2e.log &
nohup /home/ubuntu/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/LServer/rollout_lambda_5_epoch_545_vloss_2e-2.json > tasl-logs/rollout_lambda_5_epoch_545_vloss_2e.log &
nohup /home/ubuntu/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/LServer/rollout_lambda_5_epoch_545_vloss_4e-2.json > tasl-logs/lambdrollout_lambda_5_epoch_545_vloss_4e.log &

wait