"""
Implementation of Behavioral Cloning (BC).
"""
import pickle
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.distributions as D

import self_correct_robot.models.base_nets as BaseNets
import self_correct_robot.models.policy_nets as PolicyNets
import self_correct_robot.models.vae_nets as VAENets
import self_correct_robot.utils.loss_utils as LossUtils
import self_correct_robot.utils.tensor_utils as TensorUtils
import self_correct_robot.utils.torch_utils as TorchUtils
import self_correct_robot.utils.obs_utils as ObsUtils

from self_correct_robot.algo import register_algo_factory_func, PolicyAlgo
from self_correct_robot.utils.tasl_exp import get_value_target


@register_algo_factory_func("bc")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    # note: we need the check below because some configs import BCConfig and exclude
    # some of these options
    gaussian_enabled = ("gaussian" in algo_config and algo_config.gaussian.enabled)
    gmm_enabled = ("gmm" in algo_config and algo_config.gmm.enabled)
    vae_enabled = ("vae" in algo_config and algo_config.vae.enabled)

    rnn_enabled = algo_config.rnn.enabled
    transformer_enabled = algo_config.transformer.enabled

    if gaussian_enabled:
        if rnn_enabled:
            raise NotImplementedError
        elif transformer_enabled:
            raise NotImplementedError
        else:
            algo_class, algo_kwargs = BC_Gaussian, {}
    elif gmm_enabled:
        if rnn_enabled:
            algo_class, algo_kwargs = BC_RNN_GMM, {}
        elif transformer_enabled:
            algo_class, algo_kwargs = BC_Transformer_GMM, {}
        else:
            algo_class, algo_kwargs = BC_GMM, {}
    elif vae_enabled:
        if rnn_enabled:
            raise NotImplementedError
        elif transformer_enabled:
            raise NotImplementedError
        else:
            algo_class, algo_kwargs = BC_VAE, {}
    else:
        if rnn_enabled:
            algo_class, algo_kwargs = BC_RNN, {}
        elif transformer_enabled:
            algo_class, algo_kwargs = BC_Transformer, {}
        else:
            algo_class, algo_kwargs = BC, {}

    return algo_class, algo_kwargs


class BC(PolicyAlgo):
    """
    Normal BC training.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.ActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, 0, :]
        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def update_value_network(self, tau=0.005):
        for target_param, param in zip(self.target_value_model.parameters(), self.main_value_model.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def train_on_batch(self, batch, epoch, validate=False, config=None):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """

        with TorchUtils.maybe_no_grad(no_grad=validate):

            _, value_y = get_value_target(batch, config, self, self.device)

            # get action fused value, which is corresponding with the progress
            info = super(BC, self).train_on_batch(batch, epoch, validate=validate)
            predictions, value_hat = self._forward_training(batch)
            # calculate accumulated difference, if this value is greater than some threshold, re-train this model.
            losses = self._compute_losses(predictions, batch)

            value_y_delta = value_y - value_hat  # set progress to 0 to 1

            value_loss = torch.mean(value_y_delta ** 2)

            # divergence = torch.mean((value_loss - self.previous_progress_loss)**2)
            divergence = torch.mean(torch.log(self.previous_progress_loss / value_loss)**2)

            # print('divergence: ', divergence)

            loss_with_divergence = value_loss + divergence

            # epsilon = 1e-6
            # value_loss = torch.mean((torch.log(value_hat + epsilon) - torch.log(value_y + epsilon)) ** 2)

            # value_delta = get_diff_percentage(value_hat, normalized_value_y)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)
            # info[f'Parameters_hist_of_value_embedding_{epoch}'] = self.nets.policy.nets.value_embedding.weight.detach().cpu().numpy()
            info["value_loss"] = TensorUtils.detach(loss_with_divergence)

            # trust = ((100 - value_delta) ** 2) / (100 ** 2) if value_delta < 100 else 0
            # info['trust'] = TensorUtils.detach(trust).item() if not isinstance(trust, (int, float)) else trust

            info["value_lr"] = self.value_optimizer.param_groups[0]['lr']

            # tau = config.experiment.tau
            # _lambda = 0.05

            # if value_loss < self.epsilon_train:
            #     self.epsilon_train = value_loss.clone().detach()

            # epsilon_LVM = 0.1
            # vloss_threshold = 0.01
            # vloss_threshod_min = 0.02**2

            # log_prob = info["losses"]["log_probs"]
            # threshold_value = tau * (config.bias + value_loss + epsilon_LVM + self.epsilon_train)
            #
            # regularization_term = torch.max(
            #     torch.zeros_like(log_prob),
            #     threshold_value - log_prob
            # )
            #
            # info['threshold_value'] = TensorUtils.detach(threshold_value)
            # info['threshold_term'] = TensorUtils.detach(regularization_term)

            # print('regularization term: ', regularization_term.item())

            # value_loss += _lambda * regularization_term

            # beta = config.experiment.value_loss_lambda
            # value_reg_loss = beta * value_loss

            # value_loss.backward(retain_graph=True)

            self.total_step += 1

            if not validate:
                self.value_optimizer.zero_grad()
                loss_with_divergence.backward(retain_graph=True)

                gradients = []

                    # with open(f'gradient-progress-{self.total_step}.pkl', 'wb') as f:
                    #     pickle.dump(gradients, f)

                # trust_threshold = 0.80
                value_loss_threshold = 850
                # value_loss_lambda = 0.1

                torch.cuda.synchronize()  # Synchronize before moving to CPU

                value_loss_cpu = value_loss.detach().cpu().item()

                if torch.isnan(value_loss).any() or torch.isinf(value_loss).any():
                    print("value_loss contains NaN or Inf values.")
                # elif value_loss.item() < vloss_threshold:
                #     print('update policy')
                # elif value_loss_cpu < value_loss_threshold:
                    # print('updating action')
                    # losses['action_loss'] *= trust
                step_info = self._train_step(losses, self.total_step, progress_gradient=None)
                info.update(step_info)
                    # else:
                    #     torch.nn.utils.clip_grad_norm_(self.nets['policy'].parameters(), max_norm=1.0)
                    #     torch.nn.utils.clip_grad_norm_(self.nets["value_decoder"].parameters(), max_norm=1.0)

                self.value_optimizer.step()

                with torch.no_grad():
                    self.previous_progress_loss = value_loss.detach().clone()

        return info

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        predictions = OrderedDict()
        actions = self.nets["policy"](obs_dict=batch["obs"], goal_dict=batch["goal_obs"])
        predictions["actions"] = actions
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        losses = OrderedDict()
        a_target = batch["actions"]
        actions = predictions["actions"]
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        # cosine direction loss on eef delta position
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

        action_losses = [
            self.algo_config.loss.l2_weight * losses["l2_loss"],
            self.algo_config.loss.l1_weight * losses["l1_loss"],
            self.algo_config.loss.cos_weight * losses["cos_loss"],
        ]
        action_loss = sum(action_losses)
        losses["action_loss"] = action_loss
        return losses

    def _train_step(self, losses, total_step, progress_gradient):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"],
            max_grad_norm=self.global_config.train.max_grad_norm,
            total_step=total_step,
            progress_gradient=progress_gradient
        )
        info["policy_grad_norms"] = policy_grad_norms

        # step through optimizers
        for k in self.lr_schedulers:
            if self.lr_schedulers[k] is not None:
                self.lr_schedulers[k].step()
        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(BC, self).log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log['progress_loss'] = info["losses"]["progress_loss"]["action_loss"].item()
        log['expert_and_progress_diff'] = info["action_difference"].item()
        log['losses_difference'] = info["losses_difference"].item()

        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "cos_loss" in info["losses"]:
            log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training
        return self.nets["policy"](obs_dict, goal_dict=goal_dict)


class BC_Gaussian(BC):
    """
    BC training with a Gaussian policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gaussian.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GaussianActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            fixed_std=self.algo_config.gaussian.fixed_std,
            init_std=self.algo_config.gaussian.init_std,
            std_limits=(self.algo_config.gaussian.min_std, 7.5),
            std_activation=self.algo_config.gaussian.std_activation,
            low_noise_eval=self.algo_config.gaussian.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"], 
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 1
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item() 
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_GMM(BC_Gaussian):
    """
    BC training with a Gaussian Mixture Model policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.nets = self.nets.float().to(self.device)




class BC_VAE(BC):
    """
    BC training with a VAE policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.VAEActor(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            device=self.device,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **VAENets.vae_args_from_config(self.algo_config.vae),
        )
        
        self.nets = self.nets.float().to(self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Update from superclass to set categorical temperature, for categorical VAEs.
        """
        if self.algo_config.vae.prior.use_categorical:
            temperature = self.algo_config.vae.prior.categorical_init_temp - epoch * self.algo_config.vae.prior.categorical_temp_anneal_step
            temperature = max(temperature, self.algo_config.vae.prior.categorical_min_temp)
            self.nets["policy"].set_gumbel_temperature(temperature)
        return super(BC_VAE, self).train_on_batch(batch, epoch, validate=validate)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        vae_inputs = dict(
            actions=batch["actions"],
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
            freeze_encoder=batch.get("freeze_encoder", False),
        )

        vae_outputs = self.nets["policy"].forward_train(**vae_inputs)
        predictions = OrderedDict(
            actions=vae_outputs["decoder_outputs"],
            kl_loss=vae_outputs["kl_loss"],
            reconstruction_loss=vae_outputs["reconstruction_loss"],
            encoder_z=vae_outputs["encoder_z"],
        )
        if not self.algo_config.vae.prior.use_categorical:
            with torch.no_grad():
                encoder_variance = torch.exp(vae_outputs["encoder_params"]["logvar"])
            predictions["encoder_variance"] = encoder_variance
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # total loss is sum of reconstruction and KL, weighted by beta
        kl_loss = predictions["kl_loss"]
        recons_loss = predictions["reconstruction_loss"]
        action_loss = recons_loss + self.algo_config.vae.kl_weight * kl_loss
        return OrderedDict(
            recons_loss=recons_loss,
            kl_loss=kl_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["KL_Loss"] = info["losses"]["kl_loss"].item()
        log["Reconstruction_Loss"] = info["losses"]["recons_loss"].item()
        if self.algo_config.vae.prior.use_categorical:
            log["Gumbel_Temperature"] = self.nets["policy"].get_gumbel_temperature()
        else:
            log["Encoder_Variance"] = info["predictions"]["encoder_variance"].mean().item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_RNN(BC):
    """
    BC training with an RNN policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = batch["obs"]
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"]

        if self._rnn_is_open_loop:
            # replace the observation sequence with one that only consists of the first observation.
            # This way, all actions are predicted "open-loop" after the first observation, based
            # on the rnn hidden state.
            n_steps = batch["actions"].shape[1]
            obs_seq_start = TensorUtils.index_at_time(batch["obs"], ind=0)
            input_batch["obs"] = TensorUtils.unsqueeze_expand_at(obs_seq_start, size=n_steps, dim=1)

        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        if self._rnn_hidden_state is None or self._rnn_counter % self._rnn_horizon == 0:
            batch_size = list(obs_dict.values())[0].shape[0]
            self._rnn_hidden_state = self.nets["policy"].get_rnn_init_state(batch_size=batch_size, device=self.device)

            if self._rnn_is_open_loop:
                # remember the initial observation, and use it instead of the current observation
                # for open-loop action sequence prediction
                self._open_loop_obs = TensorUtils.clone(TensorUtils.detach(obs_dict))

        obs_to_use = obs_dict
        if self._rnn_is_open_loop:
            # replace current obs with last recorded obs
            obs_to_use = self._open_loop_obs

        self._rnn_counter += 1
        action, self._rnn_hidden_state = self.nets["policy"].forward_step(
            obs_to_use, goal_dict=goal_dict, rnn_state=self._rnn_hidden_state)
        return action

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self._rnn_hidden_state = None
        self._rnn_counter = 0


class BC_RNN_GMM(BC_RNN):
    """
    BC training with an RNN GMM policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.rnn.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"], 
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2 # [B, T]
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item() 
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_Transformer(BC):
    """
    BC training with a Transformer policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.transformer.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.TransformerActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.transformer_args_from_config(self.algo_config.transformer),
        )
        self._set_params_from_config()
        self.nets = self.nets.float().to(self.device)
        
    def _set_params_from_config(self):
        """
        Read specific config variables we need for training / eval.
        Called by @_create_networks method
        """
        self.context_length = self.algo_config.transformer.context_length
        self.supervise_all_steps = self.algo_config.transformer.supervise_all_steps
        self.pred_future_acs = self.algo_config.transformer.pred_future_acs
        if self.pred_future_acs:
            assert self.supervise_all_steps is True

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        h = self.context_length
        input_batch["obs"] = {k: batch["obs"][k][:, :h, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present

        if self.supervise_all_steps:
            # supervision on entire sequence (instead of just current timestep)
            if self.pred_future_acs:
                ac_start = h - 1
            else:
                ac_start = 0
            input_batch["actions"] = batch["actions"][:, ac_start:ac_start+h, :]
        else:
            # just use current timestep
            input_batch["actions"] = batch["actions"][:, h-1, :]

        if self.pred_future_acs:
            assert input_batch["actions"].shape[1] == h

        input_batch = TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        return input_batch

    def _forward_training(self, batch, epoch=None):
        """
        Internal helper function for BC_Transformer algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        # ensure that transformer context length is consistent with temporal dimension of observations
        TensorUtils.assert_size_at_dim(
            batch["obs"], 
            size=(self.context_length), 
            dim=1, 
            msg="Error: expect temporal dimension of obs batch to match transformer context length {}".format(self.context_length),
        )

        predictions = OrderedDict()
        predictions["actions"] = self.nets["policy"](obs_dict=batch["obs"], actions=None, goal_dict=batch["goal_obs"])
        if not self.supervise_all_steps:
            # only supervise final timestep
            predictions["actions"] = predictions["actions"][:, -1, :]
        return predictions

    def get_action(self, obs_dict, goal_dict=None, keep_whole=False):
        """
        Get policy action outputs.
        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal
        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        output = self.nets["policy"](obs_dict, actions=None, goal_dict=goal_dict)

        if keep_whole:
            return output
        else:
            if self.supervise_all_steps:
                if self.algo_config.transformer.pred_future_acs:
                    output = output[:, 0, :]
                else:
                    output = output[:, -1, :]
            else:
                output = output[:, -1, :]

            return output


class BC_Transformer_GMM(BC_Transformer):
    """
    BC training with a Transformer GMM policy.

    """
    def __init__(self, main_value_model, target_value_model, *args, **kwargs):
        super(BC_Transformer_GMM, self).__init__(*args, **kwargs)
        self.main_value_model = main_value_model
        self.target_value_model = target_value_model
        # self.progress_model = progress_model

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.transformer.enabled

        self.nets = nn.ModuleDict()

        # self.obs_shapes['value'] = [1]

        self.nets["policy"] = PolicyNets.TransformerGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.transformer_args_from_config(self.algo_config.transformer),
        )
        self._set_params_from_config()
        self.nets = self.nets.float().to(self.device)

        self.value_optimizer = torch.optim.Adam(
            self.nets['policy'].parameters(),
            lr=self.global_config.value_lr,
            weight_decay=self.global_config.value_weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.value_optimizer,
            mode='min',
            factor=0.75,
            patience=2,
            verbose=False
        )

        self.previous_progress_loss = 1

        self.total_step = 0

    def _forward_training(self, batch, epoch=None):
        """
        Modify from super class to support GMM training.
        """
        # ensure that transformer context length is consistent with temporal dimension of observations
        TensorUtils.assert_size_at_dim(
            batch["obs"], 
            size=(self.context_length), 
            dim=1, 
            msg="Error: expect temporal dimension of obs batch to match transformer context length {}".format(self.context_length),
        )

        dists, predict_value = self.nets["policy"].forward_train(
            obs_dict=batch["obs"], 
            actions=None,
            goal_dict=batch["goal_obs"],
            low_noise_eval=False,
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2 # [B, T]

        if not self.supervise_all_steps:
            # only use final timestep prediction by making a new distribution with only final timestep.
            # This essentially does `dists = dists[:, -1]`
            component_distribution = D.Normal(
                loc=dists.component_distribution.base_dist.loc[:, -1],
                scale=dists.component_distribution.base_dist.scale[:, -1],
            )
            component_distribution = D.Independent(component_distribution, 1)
            mixture_distribution = D.Categorical(logits=dists.mixture_distribution.logits[:, -1])
            dists = D.MixtureSameFamily(
                mixture_distribution=mixture_distribution,
                component_distribution=component_distribution,
            )

        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions, predict_value

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC_Transformer_GMM algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()

        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of info
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item()
        log['value_loss'] = info['value_loss'].item()
        # log['trust'] = info['trust']
        # log['regularization_term'] = info['threshold_term'].item()
        log['value_lr'] = info['value_lr']
        # log['threshold_value'] = info['threshold_value'].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]

        PARAMTERS_OF_VALUE_EMB = 'Parameters_hist_of_value_embedding'

        for key in info:
            if key.startswith(PARAMTERS_OF_VALUE_EMB):
                log[key] = info[key]
        return log