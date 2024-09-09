#!/bin/bash


/home/minquangao/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/tasl/tasl_xfmr_value_loss_lambda_5e-1_multi-task-with-value-correct-seed-999.json > tasl-logs/vloss-lambda-5e-1.log &
/home/minquangao/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/tasl/tasl_xfmr_multi-task-with-value-correct-seed-999.json > tasl-logs/vloss-lambda-5.log &
/home/minquangao/anaconda3/envs/robocasa/bin/python -u self_correct_robot/scripts/train.py --config=self_correct_robot/scripts/running_configs/tasl/values_loss_lambda_1e1-tasl_xfmr_multi-task-with-value-correct-seed-999.json > tasl-logs/vloss-lambda-1.log &
