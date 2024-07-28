import numpy as np

from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from tianshou.env import SubprocVectorEnv
from robomimic.utils.tasl_exp import get_value_target
from copy import deepcopy
import robomimic.utils.tensor_utils as TensorUtils


def correct_by_trust_policy_region(
    policy,
    value_model,
    config,
    env,
    horizon,
    video_write_path,
    video_writer,
    video_skip=5,
    render=False):
    """
    will execute policy in deployment phrase.

    First it will use value loss to determine if current state's action result is reliable.

    If value loss is less than some threshold, then, will use Bayesian Optimization Methods to find a new action such that

    environment can drive to a new reliable state.

    """
    assert isinstance(policy, RolloutPolicy)
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper) or isinstance(env, SubprocVectorEnv)

    batched = isinstance(env, SubprocVectorEnv)

    ob_dict = env.reset()
    policy.start_episode(lang=env._ep_lang_str)

    environment_states = []
    # save deepcopy version of environments. If we want to set backtrack, we can just reset the env's index

    # state_manager = StateManager(env)

    frame_count = 0

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

    for step_i in range(horizon):  # LogUtils.tqdm(range(horizon)):
        # get action from policy

        policy_ob = ob_dict
        policy_ob, _, _ = get_value_target(policy_ob, config, policy, policy.policy.device)
        # 1. prepare observation
        # 2. get ac, value predicated
        # 3. determine the value's loss by compared with the value_model

        rollout_prepared_batch = policy

        ac = policy(ob=policy_ob, goal=None)  # , return_ob=True)

        # play action
        ob_dict, r, done, info = env.step(ac)

        # state_manager.append(env)

        # if step_i > 20:
        #     state_manager.reverse_play_envs()
        #     print('test done!')

        # render to screen
        if render:
            env.render(mode="human")

        # compute reward
        rews.append(r)

        # cur_success_metrics = env.is_success()

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

    # TODO:if current state's normalized value is close to 1, which means almost finished the task, then break

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
        results["Success_Rate"] = float(success["task"])

    # log additional success metrics
    for k in success:
        if k != "task":
            if batched:
                results["{}_Success_Rate".format(k)] = success[k].astype(float)
            else:
                results["{}_Success_Rate".format(k)] = float(success[k])

    return results