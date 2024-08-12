import torch
from torch import nn
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.action_utils as AcUtils
from copy import deepcopy
import numpy as np
import torch.distributions as D
import torch.nn.functional as F
import cma



def concatenate_images(batch, direct_obs=False):
    # Extract the image tensors
    left_image = batch['obs']['robot0_agentview_left_image']
    right_image = batch['obs']['robot0_agentview_right_image']
    eye_in_hand_image = batch['obs']['robot0_eye_in_hand_image']

    # Concatenate the images along the width dimension (dim=3)
    concatenated_images = torch.cat((left_image, eye_in_hand_image, right_image), dim=-1)

    # if not direct_obs:
    concatenated_images = concatenated_images.permute(0, 1, 3, 4, 2)
    # else:
    #     concatenated_images = concatenated_images.permute(0, 2, 3, 1)

    # Add the new concatenated image tensor to the 'obs' dictionary
    batch['obs']['concatenated_images'] = concatenated_images

    return batch


def get_value_target(batch, config, obj, device):
    assert isinstance(batch, dict)

    direct_obs = False
    if 'obs' not in batch and 'robot0_eye_in_hand_image' in batch:
        for key in batch:
            if 'image' in key:
                batch[key] = torch.tensor(batch[key]).to(device)

        batch['obs'] = batch.copy()
        direct_obs = True
        # progress_model = obj.policy.progress_model
        # main_value_model = obj.policy.main_value_model
        target_value_model = obj.policy.target_value_model
    else:
        # progress_model = obj.progress_model
        # main_value_model = obj.main_value_model
        target_value_model = obj.target_value_model

    batch = concatenate_images(batch, direct_obs)
    obs_images = batch['obs']['concatenated_images']

    target_value_model = target_value_model.to(device)
    # progress_model = progress_model.to(device)

    # Check if obs_images has a batch dimension
    if len(obs_images.shape) == 5:
        # Case when obs_images has shape (batch, 10, 128, 384, 3)
        batch_size = obs_images.shape[0]
        step_size = obs_images.shape[1]
        reshaped_concatenated_images = obs_images.view(-1, obs_images.shape[-3], obs_images.shape[-2],
                                                       obs_images.shape[-1])
    else:
        # Case when obs_images has shape (10, 128, 384, 3)
        batch_size = 1
        step_size = obs_images.shape[0]
        reshaped_concatenated_images = obs_images.view(-1, obs_images.shape[-3], obs_images.shape[-2],
                                                       obs_images.shape[-1])

    # Move to CUDA device if available
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reshaped_concatenated_images = reshaped_concatenated_images.permute(0, 3, 1, 2).to(device)

    # Assuming progress_model, main_value_model, and target_value_model are defined and on the correct device
    # progresses = progress_model(None, reshaped_concatenated_images).view(batch_size, 10, -1)
    # p_threshold = config.progress_threshold
    # rewards = torch.where(progresses > p_threshold, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))

    target_value_model.eval()

    # value_y_hats = main_value_model(None, reshaped_concatenated_images).view(batch_size, 10, -1)
    value_y_target = target_value_model(None, reshaped_concatenated_images).view(batch_size, 10, -1)

    # value_y = torch.zeros_like(value_y_target, device=device)
    # value_y[:, 0, :] = value_y_target[:, 0, :]

    # for t in range(1, step_size):
    #     value_y[:, t, :] = value_y_target[:, t - 1, :] + rewards[:, t, :]

    # batch['obs']['value'] = value_y.view(step_size, 1) if batch_size == 1 else value_y

    # if direct_obs:
    #     batch = batch['obs']

    return batch, value_y_target


def get_diff_percentage(V, V_t):
    difference = torch.abs(V - V_t)

    # Calculate the absolute target values
    abs_target = torch.abs(V_t)

    # Avoid division by zero by adding a small epsilon where abs_target is zero
    epsilon = 1e-8
    abs_target = torch.where(abs_target == 0, torch.tensor(epsilon), abs_target)

    # Calculate the percentage difference
    percentage_difference = (difference / abs_target) * 100

    return torch.mean(percentage_difference)


def custom_init(m, lower, upper):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, lower, upper)  # Uniform initialization
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# Example of normalization function
def normalize(value, min_val=-500, max_val=1):
    return (value - min_val) / (max_val - min_val)


def denormalize(value, min_val=-500, max_val=1):
    return value * (max_val - min_val) + min_val


def post_process_ac(ac, batched, obj):
    if not batched:
        ac = ac[0]
    ac = TensorUtils.to_numpy(ac)
    if obj.action_normalization_stats is not None:
        action_keys = obj.policy.global_config.train.action_keys
        action_shapes = {k: obj.action_normalization_stats[k]["offset"].shape[1:] for k in
                         obj.action_normalization_stats}
        ac_dict = AcUtils.vector_to_action_dict(ac, action_shapes=action_shapes, action_keys=action_keys)
        ac_dict = ObsUtils.unnormalize_dict(ac_dict, normalization_stats=obj.action_normalization_stats)
        action_config = obj.policy.global_config.train.action_config
        for key, value in ac_dict.items():
            this_format = action_config[key].get("format", None)
            if this_format == "rot_6d":
                rot_6d = torch.from_numpy(value).unsqueeze(0)
                conversion_format = action_config[key].get("convert_at_runtime", "rot_axis_angle")
                if conversion_format == "rot_axis_angle":
                    rot = TorchUtils.rot_6d_to_axis_angle(rot_6d=rot_6d).squeeze().numpy()
                elif conversion_format == "rot_euler":
                    rot = TorchUtils.rot_6d_to_euler_angles(rot_6d=rot_6d, convention="XYZ").squeeze().numpy()
                else:
                    raise ValueError
                ac_dict[key] = rot
        ac = AcUtils.action_dict_to_vector(ac_dict, action_keys=action_keys)
    return ac


def get_current_state_value_loss(rollout_policy, config, obs_dict):
    obs_dict = rollout_policy._prepare_observation(obs_dict)
    tmp_ob, tmp_target_value = get_value_target(obs_dict, config, rollout_policy, rollout_policy.policy.device)
    ac_dist, value_predict = rollout_policy.policy.nets['policy'].forward_train(obs_dict=obs_dict)
    # execute_value_predict = value_predict[:, 0, :][0][0]
    # tmp_target_value = tmp_target_value[0][0][0]
    tmp_target_value = normalize(tmp_target_value)

    tmp_value_loss = torch.mean((value_predict - tmp_target_value) ** 2)

    return tmp_value_loss, ac_dist


def get_deployment_action_and_value_from_obs(rollout_policy, config, obs_dict):
    obs_dict = rollout_policy._prepare_observation(obs_dict)
    ac_dist, value_predict = rollout_policy.policy.nets['policy'].forward_train(obs_dict=obs_dict)
    execute_value_predict = value_predict[:, 0, :]
    execute_ac = ac_dist.sample()[:, 0, :]
    execute_ac = post_process_ac(execute_ac, obs_dict, obj=rollout_policy)[0]

    return ac_dist, execute_ac, execute_value_predict


AC_HAT_NEXT = None


def execute_forward_loss(ac_dist, config, env, previous_value, policy, device):
    ac = ac_dist.sample()[:, 0, :][0]

    if isinstance(ac, torch.Tensor):
        ac = ac.cpu().numpy()

    next_state_dict, _, _, _ = env.step(ac)

    next_state_dict, target_value_next = get_value_target(next_state_dict, config, policy, device)

    next_state_dict = policy._prepare_observation(next_state_dict)

    ac_hat_next, value_hat_next = policy.policy.nets['policy'].forward_train(obs_dict=next_state_dict)

    target_value_next = target_value_next.to(device)[:, 0, :][0]
    value_hat_next = value_hat_next.to(device)[:, 0, :][0]

    loss = torch.mean(((normalize(target_value_next) - value_hat_next) ** 2) + F.relu(previous_value - target_value_next))

    global AC_HAT_NEXT
    AC_HAT_NEXT = ac_hat_next

    return loss


def perturb_parameters(params, variance):
    return params + np.random.randn(*params.shape) * variance


def find_new_ac(original_ac_dist, config, env, previous_value, policy, device, threshold=0.01, num_iterations=100):

    better_ac = None

    num_iteration = 100

    original_state = deepcopy(env.get_state())

    mixture_logits = original_ac_dist.mixture_distribution.logits.detach().cpu().numpy()
    component_means = original_ac_dist.component_distribution.base_dist.loc.detach().cpu().numpy()
    component_scales = original_ac_dist.component_distribution.base_dist.scale.detach().cpu().numpy()

    params = np.concatenate([mixture_logits.flatten(), component_means.flatten(), component_scales.flatten()])

    current_params = params.copy()
    new_params = current_params

    def objective_function(params):
        # Reconstruct the MixtureSameFamily distribution from parameters
        mixture_logits_size = mixture_logits.size
        component_means_size = component_means.size
        component_scales_size = component_scales.size

        mixture_logits_new = torch.tensor(params[:mixture_logits_size], dtype=torch.float32, device=device).reshape(
            mixture_logits.shape)
        component_means_new = torch.tensor(params[mixture_logits_size:mixture_logits_size + component_means_size],
                                           dtype=torch.float32, device=device).reshape(component_means.shape)
        component_scales_new = torch.tensor(params[mixture_logits_size + component_means_size:], dtype=torch.float32,
                                            device=device).reshape(component_scales.shape)

        component_scales_new = torch.nn.functional.softplus(component_scales_new)

        mixture_dist = D.Categorical(logits=mixture_logits_new)
        component_dist = D.Independent(D.Normal(loc=component_means_new, scale=component_scales_new), 1)
        ac_dist_new = D.MixtureSameFamily(mixture_dist, component_dist)

        loss = execute_forward_loss(ac_dist_new, config, env, previous_value, policy, device)
        print('new loss = ', loss)

        return loss

    # Run CMA-ES optimization
    es = cma.CMAEvolutionStrategy(params, 0.5, {'maxiter': num_iterations})
    es.optimize(objective_function)

    # Get the best found solution and reconstruct the distribution
    best_params = es.result.xbest

    mixture_logits_best = torch.tensor(best_params[:mixture_logits.size], dtype=torch.float32, device=device).reshape(
        mixture_logits.shape)
    component_means_best = torch.tensor(best_params[mixture_logits.size:mixture_logits.size + component_means.size],
                                        dtype=torch.float32, device=device).reshape(component_means.shape)
    component_scales_best = torch.tensor(best_params[mixture_logits.size + component_means.size:], dtype=torch.float32,
                                         device=device).reshape(component_scales.shape)

    mixture_dist_best = D.Categorical(logits=mixture_logits_best)
    component_dist_best = D.Independent(D.Normal(loc=component_means_best, scale=component_scales_best), 1)
    best_ac_dist = D.MixtureSameFamily(mixture_dist_best, component_dist_best)

    better_ac = best_ac_dist.sample()[:, 0, :][0]
    better_loss = es.result.fbest
    # for i in range(num_iteration):
    #     if i > 0:
    #         env.reset_to(original_state)
    #
    #     perturbed_params = perturb_parameters(current_params, search_variance)
    #
    #     mixture_logits_size = mixture_logits.size
    #     component_means_size = component_means.size
    #     component_scales_size = component_scales.size

        # mixture_logits_new = torch.tensor(perturbed_params[:mixture_logits_size], dtype=torch.float32,
        #                                   device=device).reshape(mixture_logits.shape)
        # component_means_new = torch.tensor(
        #     perturbed_params[mixture_logits_size:mixture_logits_size + component_means_size], dtype=torch.float32,
        #     device=device).reshape(component_means.shape)
        # component_scales_new = torch.tensor(perturbed_params[mixture_logits_size + component_means_size:],
        #                                     dtype=torch.float32, device=device).reshape(component_scales.shape)
        #
        # component_scales_new = torch.nn.functional.softplus(component_scales_new)


        # mixture_dist_new = D.Categorical(logits=mixture_logits_new)
        # component_dist_new = D.Independent(D.Normal(loc=component_means_new, scale=component_scales_new), 1)
        # perturbed_ac_dist = D.MixtureSameFamily(mixture_dist_new, component_dist_new)

        # perturbed_loss, ac = execute_forward_loss(perturbed_ac_dist, config, env, previous_value, policy, device)

        # print(f'trying iter: {i} get loss {perturbed_loss}')

        # if perturbed_loss < threshold:
        #     better_ac = ac
        #     print('find new ac dist')

    return better_ac, better_loss




