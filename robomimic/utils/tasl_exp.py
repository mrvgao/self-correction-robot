import torch


def concatenate_images(batch, direct_obs=False):
    # Extract the image tensors
    left_image = batch['obs']['robot0_agentview_left_image']
    right_image = batch['obs']['robot0_agentview_right_image']
    eye_in_hand_image = batch['obs']['robot0_eye_in_hand_image']

    # Concatenate the images along the width dimension (dim=3)
    concatenated_images = torch.cat((left_image, eye_in_hand_image, right_image), dim=-1)

    if not direct_obs:
        concatenated_images = concatenated_images.permute(0, 1, 3, 4, 2)
    else:
        concatenated_images = concatenated_images.permute(0, 2, 3, 1)

    # Add the new concatenated image tensor to the 'obs' dictionary
    batch['obs']['concatenated_images'] = concatenated_images

    return batch


def add_value(batch, config, obj, device):
    assert isinstance(batch, dict)

    direct_obs = False
    if 'obs' not in batch and 'robot0_eye_in_hand_image' in batch:
        for key in batch:
            if 'image' in key:
                batch[key] = torch.tensor(batch[key]).to(device)

        batch['obs'] = batch.copy()
        direct_obs = True
        progress_model = obj.policy.progress_model
        main_value_model = obj.policy.main_value_model
        target_value_model = obj.policy.target_value_model
    else:
        progress_model = obj.progress_model
        main_value_model = obj.main_value_model
        target_value_model = obj.target_value_model

    batch = concatenate_images(batch, direct_obs)
    obs_images = batch['obs']['concatenated_images']

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reshaped_concatenated_images = reshaped_concatenated_images.permute(0, 3, 1, 2).to(device)

    # Assuming progress_model, main_value_model, and target_value_model are defined and on the correct device
    progresses = progress_model(None, reshaped_concatenated_images).view(batch_size, 10, -1)
    p_threshold = config.progress_threshold
    rewards = torch.where(progresses > p_threshold, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))

    value_y_hats = main_value_model(None, reshaped_concatenated_images).view(batch_size, 10, -1)
    value_y_target = target_value_model(None, reshaped_concatenated_images).view(batch_size, 10, -1)

    value_y = torch.zeros_like(value_y_target, device=device)
    value_y[:, 0, :] = value_y_target[:, 0, :]

    for t in range(1, step_size):
        value_y[:, t, :] = value_y_target[:, t - 1, :] + rewards[:, t, :]

    batch['obs']['value'] = value_y.view(step_size, 1) if batch_size == 1 else value_y

    if direct_obs:
        batch = batch['obs']

    return batch, value_y_hats, value_y_target
