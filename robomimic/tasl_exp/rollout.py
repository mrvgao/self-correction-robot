from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy
from tianshou.env import SubprocVectorEnv
from robomimic.utils.exp_utils import StateManager
from robomimic.utils.tasl_exp import  normalize, get_deployment_action_and_value_from_obs, concatenate_images, get_value_target
from robomimic.utils.tasl_exp import find_new_ac, post_process_ac
import json
import imageio
from collections import OrderedDict
from copy import deepcopy
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.log_utils as LogUtils
import numpy as np
import os
import time
import traceback
import torch
import cv2
import pickle
from tqdm import tqdm
from robomimic.utils.tasl_exp import get_current_state_value_loss
import copy


def adaptive_threshold(i, max_step):
    start = 0.011
    end = 0.015
    threshold = start + ((end - start) / max_step) * i
    return threshold


def find_reliable_action(step_i, ob_dict, env, policy, config, video_frames):
    original_state = env.get_state()
    TRYING = 10

    tmp_value_loss_current, ac_dist = get_current_state_value_loss(policy, config, ob_dict)

    max_trust = -float('inf')

    # max_ac = None
    find = False

    minimal_loss = float('inf')
    minimal_loss_state = None
    minimal_loss_ac_dist = None
    minimal_index = -1

    previous_value = -float('inf')

    # CONSTRAINT_FORWARD = False

    THRESHOLD = 1e-3

    trying = 0
    TRYING_MAX = 50

    find = tmp_value_loss_current < THRESHOLD

    while not find and trying < TRYING_MAX:
        policy.policy.nets['policy'].train()
        policy.policy.value_optimizer.zero_grad()
        tmp_value_loss_current.backward()
        policy.policy.value_optimizer.step()
        tmp_value_loss_current, ac_dist = get_current_state_value_loss(policy, config, ob_dict)
        find = tmp_value_loss_current < THRESHOLD
        trying += 1
        # tqdm.write(f'trying, {trying}/{TRYING_MAX}, loss is {tmp_value_loss_current}')

    policy.policy.nets['policy'].eval()

    return find

    # for i in range(TRYING):
    #
    #     trust_threshold = adaptive_threshold(i, TRYING-1)
    #
    #
    #     if step_i == 0:
    #         # tmp_value_loss = tmp_value_loss_current
    #         if tmp_value_loss_current < minimal_loss:
    #             minimal_loss = tmp_value_loss_current
    #             minimal_loss_state = env.get_state()
    #             minimal_loss_ac_dist = ac_dist
    #             minimal_index = i
    #
    #         print(f'trying time: {i}, miniaml loss is :{minimal_loss} threshold is : {trust_threshold}')
    #
    #         if minimal_loss > trust_threshold:
    #             ob_dict = env.reset()
    #             tmp_value_loss_current, ac_dist = get_current_state_value_loss(policy, config, ob_dict)
    #         else:
    #             if minimal_index != i: env = env.reset_to(minimal_loss_state)
    #
    #             frame = env.render(mode="rgb_array", height=512, width=512)
    #             video_frames.append(frame)
    #             find = True
    #             break
    #     else:
    #         sample = ac_dist.sample()
    #         tmp_ac = sample[:, 0, :]
    #
    #         # apply ac
    #         tmp_ac = post_process_ac(tmp_ac, False, obj=policy)
    #         tmp_ob_dict_next, _, _, _ = env.step(tmp_ac)  # drive first time
    #
    #         _, tmp_target_value_next = get_value_target(tmp_ob_dict_next, config, policy, policy.policy.device)
    #
    #         tmp_prepared_batch_next = policy._prepare_observation(tmp_ob_dict_next)
    #         tmp_next_ac_dist, tmp_value_next = policy.policy.nets['policy'].forward_train(obs_dict=tmp_prepared_batch_next)
    #
    #         tmp_target_value = tmp_target_value_next
    #         tmp_target_value = normalize(tmp_target_value)
    #
    #         tmp_value = tmp_value_next
    #         #
    #         tmp_value_loss_forward = torch.mean((tmp_target_value - tmp_value)**2)
    #
    #         if tmp_value_loss_forward < minimal_loss:
    #             minimal_loss = tmp_value_loss_forward
    #             minimal_loss_state = env.get_state()
    #             minimal_loss_ac_dist = ac_dist
    #             minimal_index = i
    #
    #         print(f'trying time: {i}, mini loss is :{minimal_loss} threshold is : {trust_threshold}')
    #
    #         if minimal_loss < trust_threshold: # get the action that can drive to next state
    #             print('find NEW action that can drive to TRUST state')
    #             find = True
    #             if i != minimal_index: env.reset_to(minimal_loss_state)
    #
    #             revert_frame = env.render(mode="rgb_array", height=512, width=512)
    #             video_frames.append(revert_frame)
    #             break
    #         else:
    #             env.reset_to(original_state)
    #
    # return find


def run_rollout(
        policy,
        env,
        horizon,
        use_goals=False,
        render=False,
        video_writer=None,
        video_skip=5,
        terminate_on_success=False,
        frame_save_dir=None,
        config=None,
        device=None,
        value_model=None,
        with_progress_correct=False,
        progress_bar=None
):
    """
    Runs a rollout in an environment with the current network parameters.

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        env (EnvBase instance): environment to use for rollouts.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        render (bool): if True, render the rollout to the screen

        video_writer (imageio Writer instance): if not None, use video writer object to append frames at
            rate given by @video_skip

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

    Returns:
        results (dict): dictionary containing return, success rate, etc.
    """
    assert isinstance(policy, RolloutPolicy)
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper) or isinstance(env, SubprocVectorEnv)

    batched = isinstance(env, SubprocVectorEnv)

    ob_dict = env.reset()
    policy.start_episode(lang=env._ep_lang_str)

    environment_name = env._env_name

    goal_dict = None
    if use_goals:
        # retrieve goal from the environment
        goal_dict = env.get_goal()

    results = {}
    video_count = 0  # video frame counter

    rews = []
    success = None  # { k: False for k in env.is_success() } # success metrics

    if batched:
        end_step = [None for _ in range(len(env))]
    else:
        end_step = None

    if batched:
        video_frames = [[] for _ in range(len(env))]
    else:
        video_frames = []

    STATE, LOSS = 'states', 'loss'
    abnormal_states = {STATE: [], LOSS: []}

    final_step = 0

    previous_states = env.get_state()

    step_i = 0

    # while step_i < config.experiment.rollout.horizon:
    previous_gripper_pose = None

    for step_i in range(config.experiment.rollout.horizon):

        # print('step := {}/{}'.format(step_i, horizon))

        final_step = step_i

        if with_progress_correct:
            # original_ac_dist, execute_ac, execute_value_predict = get_deployment_action_and_value_from_obs(
            #     rollout_policy=policy, obs_dict=ob_dict)
            find = find_reliable_action(step_i, ob_dict, env, policy, config, video_frames)
            # tmp_value_loss_current, ac_dist = get_current_state_value_loss(policy, config, ob_dict)
            # print('tmp value loss', tmp_value_loss_current)

            # loss_threshold = 0.01
            #
            # if step_i == 0:
            #     for i in range(10):
            #         if tmp_value_loss_current > loss_threshold:
            #             print(f'trying: {i} more with ', tmp_value_loss_current)
            #             ob_dict = env.reset()
            #             tmp_value_loss_current, ac_dist = get_current_state_value_loss(policy, config, ob_dict)
            #         else:
            #             step_i += 1

            # if step_i > 0 and tmp_value_loss_current > loss_threshold:
            #     env.reset_to(previous_states)
            # elif step_i > 0 and tmp_value_loss_current < loss_threshold:
            #     step_i += 1

            if not find:
                abnormal_states[STATE].append(env.get_state())
                print('cannot find reliable forward state!')
                # abnormal_states[LOSS].append(current_value_loss)
                # print('we cannot find a new action that can drive to trust state')
                # print('re-start a new task')
                break
            # else:
                # previous_value = target_value
                # break
                # print('this state is reliable!')
        # else:
            # tmp_value_loss_current, ac_dist = get_current_state_value_loss(policy, config, ob_dict)
            # print('tmp value loss', tmp_value_loss_current)
            previous_states = env.get_state()
        ac = policy(ob=ob_dict, goal=goal_dict)
        ob_dict, r, done, _ = env.step(ac)
        progress_bar.update(1)

        GRIPPER_KEY = 'robot0_gripper_qpos'

        if previous_gripper_pose is not None:
            delta = np.mean(ob_dict[GRIPPER_KEY] - previous_gripper_pose)
            if delta == 0:
                print('FINISH THIS TASK!')
                break

        previous_gripper_pose = ob_dict[GRIPPER_KEY]

        # rews.append(r)

        # cur_success_metrics = env.is_success()
        # if batched:
        #     cur_success_metrics = TensorUtils.list_of_flat_dict_to_dict_of_list(
        #         [info[i]["is_success"] for i in range(len(info))])
        #     cur_success_metrics = {k: np.array(v) for (k, v) in cur_success_metrics.items()}
        # else:
        #     cur_success_metrics = info["is_success"]

        # if success is None:
        #     success = deepcopy(cur_success_metrics)
        # else:
        #     for k in success:
        #         success[k] = success[k] | cur_success_metrics[k]

        # visualization
        if video_writer is not None:
            if video_count % video_skip == 0:
                if batched:
                    # frames = env.render(mode="rgb_array", height=video_height, width=video_width)

                    frames = []
                    policy_ob = deepcopy(policy_ob)
                    for env_i in range(len(env)):
                        cam_imgs = []
                        for im_name in ["robot0_agentview_left_image", "robot0_agentview_right_image",
                                        "robot0_eye_in_hand_image"]:
                            im = TensorUtils.to_numpy(
                                policy_ob[im_name][env_i, -1]
                            )
                            im = np.transpose(im, (1, 2, 0))
                            if policy_ob.get("ret", None) is not None:
                                im_ret = TensorUtils.to_numpy(
                                    policy_ob["ret"]["obs"][im_name][env_i, :, -1]
                                )
                                im_ret = np.transpose(im_ret, (0, 2, 3, 1))
                                im = np.concatenate((im, *im_ret), axis=0)
                            cam_imgs.append(im)
                        frame = np.concatenate(cam_imgs, axis=1)
                        frame = (frame * 255.0).astype(np.uint8)
                        frames.append(frame)

                    for env_i in range(len(env)):
                        frame = frames[env_i]
                        video_frames[env_i].append(frame)
                else:
                    frame = env.render(mode="rgb_array", height=512, width=512)

                    # cam_imgs = []
                    # for im_name in ["robot0_eye_in_hand_image", "robot0_agentview_right_image", "robot0_agentview_left_image"]:
                    #     im_input = TensorUtils.to_numpy(
                    #         policy_ob_dict[im_name][0,-1]
                    #     )
                    #     im_ret = TensorUtils.to_numpy(
                    #         policy_ob_dict["ret"]["obs"][im_name][0,:,-1]
                    #     )
                    #     im_input = np.transpose(im_input, (1, 2, 0))
                    #     im_input = add_border_to_frame(im_input, border_size=3, color="black")
                    #     im_ret = np.transpose(im_ret, (0, 2, 3, 1))
                    #     im = np.concatenate((im_input, *im_ret), axis=1)
                    #     cam_imgs.append(im)

                    # frame = np.concatenate(cam_imgs, axis=0)
                    # image_left = ob_dict['robot0_agentview_left_image']
                    # image_hand = ob_dict['robot0_eye_in_hand_image']
                    # image_right = ob_dict['robot0_agentview_right_image']

                    # for left, hand, right in zip(image_left, image_hand, image_right):
                    #     left = left.transpose((1, 2, 0))
                    #     right = right.transpose((1, 2, 0))
                    #     hand = hand.transpose((1, 2, 0))
                    #
                    #     whole_image = np.concatenate((left, hand, right), axis=1) * 255
                    #     whole_image = whole_image.astype(np.uint8)
                    #     whole_image = cv2.cvtColor(whole_image, cv2.COLOR_BGR2RGB)
                    #     frame_count += 1
                    # save_path = os.path.join(frame_save_dir, str(frame_count) + '.png')
                    # cv2.imwrite(save_path, whole_image)
                    video_frames.append(frame)

            video_count += 1

        # break if done
        if batched:
            for env_i in range(len(env)):
                if end_step[env_i] is not None:
                    continue

                # if done[env_i] or (terminate_on_success and success["task"][env_i]):
                #     end_step[env_i] = step_i
        # else:
        #     if done or (terminate_on_success and success["task"]):
        #         end_step = step_i
        #         break

    with open(f'abnormal_states_{frame_save_dir}.pkl', 'wb') as f:
        pickle.dump(abnormal_states, f)
        # print('PICKLED abnormal state done!, LAST STEP IS', final_step)

    if video_writer is not None:
        if batched:
            for env_i in range(len(video_frames)):
                for frame in video_frames[env_i]:
                    video_writer.append_data(frame)
        else:
            for frame in video_frames:
                video_writer.append_data(frame)

    if batched:
        total_reward = np.zeros(len(env))
        rews = np.array(rews)
        for env_i in range(len(env)):
            end_step_env_i = end_step[env_i] or step_i
            total_reward[env_i] = np.sum(rews[:end_step_env_i + 1, env_i])
            end_step[env_i] = end_step_env_i

        results["Return"] = total_reward
        results["Horizon"] = np.array(end_step) + 1
        results["Success_Rate"] = success["task"].astype(float)
    else:
        end_step = end_step or step_i
        total_reward = np.sum(rews[:end_step + 1])

        results["Return"] = total_reward
        results["Horizon"] = end_step + 1
        # results["Success_Rate"] = float(success["task"])

    # log additional success metrics
    # for k in success:
    #     if k != "task":
    #         if batched:
    #             results["{}_Success_Rate".format(k)] = success[k].astype(float)
    #         else:
    #             results["{}_Success_Rate".format(k)] = float(success[k])

    results['final_step'] = final_step

    return results


def rollout_with_stats(
        policy,
        envs,
        horizon,
        use_goals=False,
        num_episodes=None,
        render=False,
        video_dir=None,
        video_path=None,
        epoch=None,
        video_skip=5,
        terminate_on_success=False,
        verbose=False,
        del_envs_after_rollouts=False,
        data_logger=None,
        config=None,
        device=None,
        value_model=None,
        with_progress_correct=False
):
    """
    A helper function used in the train loop to conduct evaluation rollouts per environment
    and summarize the results.

    Can specify @video_dir (to dump a video per environment) or @video_path (to dump a single video
    for all environments).

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        envs (dict): dictionary that maps env_name (str) to EnvBase instance. The policy will
            be rolled out in each env.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        num_episodes (int): number of rollout episodes per environment

        render (bool): if True, render the rollout to the screen

        video_dir (str): if not None, dump rollout videos to this directory (one per environment)

        video_path (str): if not None, dump a single rollout video for all environments

        epoch (int): epoch number (used for video naming)

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

        verbose (bool): if True, print results of each rollout

    Returns:
        all_rollout_logs (dict): dictionary of rollout statistics (e.g. return, success rate, ...)
            averaged across all rollouts

        video_paths (dict): path to rollout videos for each environment
    """
    assert isinstance(policy, RolloutPolicy)

    all_rollout_logs = OrderedDict()

    # handle paths and create writers for video writing
    assert (video_path is None) or (video_dir is None), "rollout_with_stats: can't specify both video path and dir"
    write_video = (video_path is not None) or (video_dir is not None)

    if isinstance(horizon, list):
        horizon_list = horizon
    else:
        horizon_list = [horizon]

    for env, horizon in zip(envs, horizon_list):
        batched = isinstance(env, SubprocVectorEnv)

        if batched:
            env_name = env.get_env_attr(key="name", id=0)[0]
        else:
            env_name = env.name

        if video_dir is not None:
            # video is written per env
            video_str = "_epoch_{}.mp4".format(epoch) if epoch is not None else ".mp4"
            video_path = os.path.join(video_dir, "{}{}".format(env_name, video_str))
            video_writer = imageio.get_writer(video_path, fps=20)

        env_video_writer = None
        if write_video:
            print("video writes to " + video_path)
            env_video_writer = imageio.get_writer(video_path, fps=20)

        print("rollout: env={}, horizon={}, use_goals={}, num_episodes={}".format(
            env_name, horizon, use_goals, num_episodes,
        ))
        rollout_logs = []
        if batched:
            iterator = range(0, num_episodes, len(env))
        else:
            iterator = range(num_episodes)

        # if not verbose:
        #     iterator = LogUtils.custom_tqdm(iterator, total=num_episodes)

        num_success = 0

        save_frames_dir_base = 'roll_out_saved_frames'

        final_steps = []

        with tqdm(total=num_episodes*config.experiment.rollout.horizon, desc='rollout progress') as pbar:
            for ep_i in iterator:
                rollout_timestamp = time.time()

                save_frames_dir = save_frames_dir_base + '-' + str(ep_i)
                # if not os.path.exists(save_frames_dir):
                #     os.makedirs(save_frames_dir)

                try:
                    previous_policy_parameters = copy.deepcopy(policy.policy.nets['policy'].state_dict())

                    rollout_info = run_rollout(
                        policy=policy,
                        env=env,
                        horizon=horizon,
                        render=render,
                        use_goals=use_goals,
                        video_writer=env_video_writer,
                        video_skip=video_skip,
                        terminate_on_success=terminate_on_success,
                        frame_save_dir=save_frames_dir,
                        config=config,
                        device=device,
                        value_model=value_model,
                        with_progress_correct=with_progress_correct,
                        progress_bar=pbar
                    )

                    policy.policy.nets['policy'].load_state_dict(previous_policy_parameters)

                    final_steps.append(rollout_info['final_step'])

                except Exception as e:
                    print("Rollout exception at episode number {}!".format(ep_i))
                    print(traceback.format_exc())
                    break

            if batched:
                rollout_info["time"] = [(time.time() - rollout_timestamp) / len(env)] * len(env)

                for env_i in range(len(env)):
                    rollout_logs.append({k: rollout_info[k][env_i] for k in rollout_info})
                num_success += np.sum(rollout_info["Success_Rate"])
            else:
                rollout_info["time"] = time.time() - rollout_timestamp

                rollout_logs.append(rollout_info)
                # num_success += rollout_info["Success_Rate"]

            if verbose:
                if batched:
                    raise NotImplementedError
                print("Episode {}, horizon={}, num_success={}".format(ep_i + 1, horizon, num_success))
                print(json.dumps(rollout_info, sort_keys=True, indent=4))

        if video_dir is not None:
            # close this env's video writer (next env has it's own)
            env_video_writer.close()

        # average metric across all episodes
        if len(rollout_logs) > 0:
            rollout_logs = dict((k, [rollout_logs[i][k] for i in range(len(rollout_logs))]) for k in rollout_logs[0])
            rollout_logs_mean = dict((k, np.mean(v)) for k, v in rollout_logs.items())
            rollout_logs_mean["Time_Episode"] = np.sum(
                rollout_logs["time"]) / 60.  # total time taken for rollouts in minutes
            all_rollout_logs[env_name] = rollout_logs_mean
        else:
            all_rollout_logs[env_name] = {"Time_Episode": -1, "Return": -1, "Success_Rate": -1, "time": -1}

        if del_envs_after_rollouts:
            # delete the environment after use
            del env

        if data_logger is not None:
            # summarize results from rollouts to tensorboard and terminal
            rollout_logs = all_rollout_logs[env_name]
            for k, v in rollout_logs.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
                else:
                    data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)

            print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
            print('Env: {}'.format(env_name))
            print(json.dumps(rollout_logs, sort_keys=True, indent=4))

    if video_path is not None:
        # close video writer that was used for all envs
        video_writer.close()

    return all_rollout_logs, None



def should_save_from_rollout_logs(
        all_rollout_logs,
        best_return,
        best_success_rate,
        epoch_ckpt_name,
        save_on_best_rollout_return,
        save_on_best_rollout_success_rate,
):
    """
    Helper function used during training to determine whether checkpoints and videos
    should be saved. It will modify input attributes appropriately (such as updating
    the best returns and success rates seen and modifying the epoch ckpt name), and
    returns a dict with the updated statistics.

    Args:
        all_rollout_logs (dict): dictionary of rollout results that should be consistent
            with the output of @rollout_with_stats

        best_return (dict): dictionary that stores the best average rollout return seen so far
            during training, for each environment

        best_success_rate (dict): dictionary that stores the best average success rate seen so far
            during training, for each environment

        epoch_ckpt_name (str): what to name the checkpoint file - this name might be modified
            by this function

        save_on_best_rollout_return (bool): if True, should save checkpoints that achieve a
            new best rollout return

        save_on_best_rollout_success_rate (bool): if True, should save checkpoints that achieve a
            new best rollout success rate

    Returns:
        save_info (dict): dictionary that contains updated input attributes @best_return,
            @best_success_rate, @epoch_ckpt_name, along with two additional attributes
            @should_save_ckpt (True if should save this checkpoint), and @ckpt_reason
            (string that contains the reason for saving the checkpoint)
    """
    should_save_ckpt = False
    ckpt_reason = None
    for env_name in all_rollout_logs:
        rollout_logs = all_rollout_logs[env_name]

        if rollout_logs["Return"] > best_return[env_name]:
            best_return[env_name] = rollout_logs["Return"]
            if save_on_best_rollout_return:
                # save checkpoint if achieve new best return
                epoch_ckpt_name += "_{}_return_{}".format(env_name, best_return[env_name])
                should_save_ckpt = True
                ckpt_reason = "return"

        # if rollout_logs["Success_Rate"] > best_success_rate[env_name]:
        #     best_success_rate[env_name] = rollout_logs["Success_Rate"]
        #     if save_on_best_rollout_success_rate:
        #         # save checkpoint if achieve new best success rate
        #         epoch_ckpt_name += "_{}_success_{}".format(env_name, best_success_rate[env_name])
        #         should_save_ckpt = True
        #         ckpt_reason = "success"

    # return the modified input attributes
    return dict(
        best_return=best_return,
        best_success_rate=best_success_rate,
        epoch_ckpt_name=epoch_ckpt_name,
        should_save_ckpt=should_save_ckpt,
        ckpt_reason=ckpt_reason,
    )

