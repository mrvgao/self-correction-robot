import torch


def concatenate_images(batch):
    # Extract the image tensors
    left_image = batch['obs']['robot0_agentview_left_image']
    right_image = batch['obs']['robot0_agentview_right_image']
    eye_in_hand_image = batch['obs']['robot0_eye_in_hand_image']

    # Concatenate the images along the width dimension (dim=3)
    concatenated_images = torch.cat((left_image, eye_in_hand_image, right_image), dim=4)
    concatenated_images = concatenated_images.permute(0, 1, 3, 4, 2)

    # Add the new concatenated image tensor to the 'obs' dictionary
    batch['obs']['concatenated_images'] = concatenated_images

    return batch
