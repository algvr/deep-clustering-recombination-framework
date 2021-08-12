import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms

import utils


class DesignPattern:
    def __init__(self, method, pattern_cfg, phase_cfg,
                 dataset_cfg, dataset_num_samples, dataset_loader_construction_function):
        # assumes 'pattern_config' has already been validated
        self.method = method
        self.config = pattern_cfg
        self.phase_cfg = phase_cfg
        self.dataset_cfg = dataset_cfg
        self.dataset_num_samples = dataset_num_samples
        self.dataset_loader_construction_function = dataset_loader_construction_function
        self.pattern_name = self.config['pattern']
        self.loss_report_interval = self.config.get('loss_report_interval', -1)
        self.running_losses = {}
        self.iteration_losses_grouped = {}  # grouped by optimizer groups
        self.loss_optimizer_groups = {}
        self.batch_augmentation_names = []
        # note: subclasses will not proceed with initialization
        # if initialization_success gets set to False during super().__init__() call
        self.initialization_success = True
        self.initialization_err_msg = ''

    def report_losses(self, new_losses, epoch_num, epoch_iter_num, phase_iter_num):
        for loss_name in new_losses:
            loss_group_name = self.loss_optimizer_groups.get(loss_name, 'default')
            if loss_group_name not in self.iteration_losses_grouped:
                self.iteration_losses_grouped[loss_group_name] = torch.zeros(1, device=self.method.device)
            self.iteration_losses_grouped[loss_group_name] += new_losses[loss_name]

        num_losses = len(self.running_losses)
        if num_losses == 0:
            return

        if self.loss_report_interval > 0:
            if phase_iter_num % self.loss_report_interval == 0:
                if num_losses == 1:
                    single_loss_name = next(iter(self.running_losses))
                    self.method.log('Running loss for "%s" at epoch %i, iteration %i: %.3f' %
                                    (self.pattern_name, epoch_num, epoch_iter_num,
                                     self.running_losses[single_loss_name].cpu().item() /
                                     self.loss_report_interval))
                    self.running_losses[single_loss_name].zero_()
                else:
                    for loss_name in self.running_losses:
                        self.method.log('Running loss "%s" for "%s" at epoch %i, iteration %i: %.3f' %
                                        (loss_name, self.pattern_name, epoch_num, epoch_iter_num,
                                         self.running_losses[loss_name].cpu().item() /
                                         self.loss_report_interval))
                        self.running_losses[loss_name].zero_()
            else:
                for loss_name in self.running_losses:
                    if loss_name in new_losses:
                        self.running_losses[loss_name] += new_losses[loss_name].detach()

    # used to allow overriding of default cluster assignment strategy in order to train individual classifiers
    # when working with multiple classifiers
    def get_cluster_assignments(self, samples, epoch_num, epoch_iter_num, phase_iter_num, try_get_logits):
        if 'classifier_name' in self.config:
            classifier = next(filter(lambda c: c.name == self.config['classifier_name'], self.method.mappings), None)
            if classifier is None:
                return None
            classification_function = self.method.get_classification_function_from_classifier(classifier)
            return None if classification_function is None else classification_function(samples)
        return self.method.get_cluster_assignments(samples, self.dataset_cfg['num_clusters'],
                                                   epoch_num, epoch_iter_num, phase_iter_num,
                                                   self.method.current_centroids,
                                                   self.method.dataset_classifier_performances,
                                                   try_get_logits)

    # returns: success, err_msg
    def exit_phase(self):
        return True, ''

    # returns: success, err_msg
    def iteration(self, epoch_num, epoch_iter_num, phase_iter_num, batch_samples, batch_labels):
        self.iteration_losses_grouped.clear()
        return True, ''


class TrainingFeatureExtractorThroughReconstructionOfSamples(DesignPattern):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.initialization_success:
            return

        self.encoder = next(filter(lambda c: c.name == self.config['encoder_name'], self.method.mappings), None)
        if self.encoder is None:
            self.initialization_success, self.initialization_err_msg = \
                False, 'Error: could not find encoder for design pattern "%s"' % self.pattern_name
            return

        self.decoder = next(filter(lambda c: c.name == self.config['decoder_name'], self.method.mappings), None)
        if self.decoder is None:
            self.initialization_success, self.initialization_err_msg = \
                False, 'Error: could not find decoder for design pattern "%s"' % self.pattern_name
            return

        self.reduction_method = self.config.get('loss_reduction_method', 'mean_across_samples')

        self.reconstruction_loss_weight = self.config.get('loss_weight', 1.0)
        self.running_losses['reconstruction_loss'] = torch.zeros(1, requires_grad=False, device=self.method.device)
        self.loss_optimizer_groups['reconstruction_loss'] = self.config.get('loss_optimizer_group_name', 'default')

        self.batch_augmentation_names = [self.config['batch_augmentation_name']] \
            if 'batch_augmentation_name' in self.config else []

    def iteration(self, epoch_num, epoch_iter_num, phase_iter_num, batch_samples, batch_labels):
        success, err_msg = super().iteration(epoch_num, epoch_iter_num, phase_iter_num, batch_samples, batch_labels)
        if not success:
            return success, err_msg

        feature_representations = self.encoder(batch_samples)
        reconstructions = self.decoder(feature_representations)

        batch_size = batch_samples.shape[0]
        unweighted_loss = None
        if self.reduction_method == 'mean_across_elements':
            unweighted_loss = F.mse_loss(batch_samples, reconstructions, reduction='mean')
        elif self.reduction_method == 'mean_across_samples':
            unweighted_loss = ((batch_samples - reconstructions) ** 2.0).sum() / batch_size
        elif self.reduction_method == 'sum':
            self.reduction_method = ((batch_samples - reconstructions) ** 2.0).sum()

        reconstruction_loss = self.reconstruction_loss_weight * unweighted_loss

        self.report_losses({'reconstruction_loss': reconstruction_loss}, epoch_num, epoch_iter_num, phase_iter_num)
        return True, ''


class EncouragingClusterFormationByMinDivergenceBetweenCurrentAndTargetClusterAssignmentDistr(DesignPattern):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.initialization_success:
            return

        if 'shuffle' in self.dataset_cfg and self.dataset_cfg['shuffle']:
            self.initialization_success = False
            self.initialization_err_msg = 'The "%s" design pattern cannot be used if shuffle=True on a dataset' \
                                          % self.pattern_name
            return

        self.target_distribution_recalculation_interval = self.config['target_distribution_recalculation_interval']
        self.target_distribution = None

        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        self.divergence_loss_weight = self.config.get('sloss_weight', 1.0)
        self.running_losses['divergence_loss'] = torch.zeros(1, requires_grad=False, device=self.method.device)
        self.loss_optimizer_groups['divergence_loss'] = self.config.get('loss_optimizer_group_name', 'default')

        self.batch_augmentation_names = [self.config['batch_augmentation_name']] \
            if 'batch_augmentation_name' in self.config else []

    def iteration(self, epoch_num, epoch_iter_num, phase_iter_num, batch_samples, batch_labels):
        success, err_msg = super().iteration(epoch_num, epoch_iter_num, phase_iter_num, batch_samples, batch_labels)
        if not success:
            return success, err_msg

        # recalculate target distribution
        if phase_iter_num % self.target_distribution_recalculation_interval == 0 or self.target_distribution is None:
            dataset_loader = self.dataset_loader_construction_function(batch_size=self.dataset_num_samples)
            all_samples, _ = next(iter(dataset_loader))
            all_samples = all_samples.to(self.method.device)
            current_cluster_assignments = \
                self.get_cluster_assignments(all_samples, epoch_num, epoch_iter_num, phase_iter_num,
                                             try_get_logits=False)

            self.target_distribution = torch.mul(current_cluster_assignments, current_cluster_assignments)
            f_js = torch.sum(current_cluster_assignments, dim=0) \
                .repeat(self.dataset_num_samples, 1).view(self.dataset_num_samples, -1)
            self.target_distribution.mul_(1.0 / f_js)

            target_distribution_norm = torch.sum(self.target_distribution, dim=1) \
                .repeat_interleave(self.dataset_cfg['num_clusters'], dim=0).view(self.dataset_num_samples, -1)
            self.target_distribution.mul_(1.0 / target_distribution_norm)
            self.target_distribution.detach_()

        batch_cluster_assignments = \
            self.get_cluster_assignments(batch_samples, epoch_num, epoch_iter_num, phase_iter_num,
                                         try_get_logits=True)

        # see constructor: this design pattern cannot be used if shuffle=True on a dataset
        pos = (epoch_iter_num - 1) * self.dataset_cfg['batch_size']
        target_distribution_slice = self.target_distribution[pos:pos + self.dataset_cfg['batch_size'], ]

        divergence_loss = \
            self.divergence_loss_weight * \
            self.kl_div_loss(F.log_softmax(batch_cluster_assignments, dim=-1), target_distribution_slice)

        self.report_losses({'divergence_loss': divergence_loss}, epoch_num, epoch_iter_num, phase_iter_num)
        return True, ''


class LearningFeatureRepresentationsByUsingContrastiveLearningAndDataAugmentation(DesignPattern):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.initialization_success:
            return

        self.contrastive_learning_head = next(filter(lambda c: c.name == self.config['contrastive_learning_head_name'],
                                                     self.method.mappings), None)
        if self.contrastive_learning_head is None:
            self.initialization_success = False
            self.initialization_err_msg = 'Error: contrastive learning head "%s" for design pattern "%s" not found ' % \
                                          (self.config['contrastive_learning_head_name'], self.pattern_name)
            return

        self.temperature_parameter = self.config.get('temperature_parameter', 0.5)
        self.cos_sim = nn.CosineSimilarity(dim=1)

        batch_size = self.dataset_cfg['batch_size']
        self.diagonal_filter = \
            (torch.ones((batch_size, batch_size), device=self.method.device) -
             torch.eye(batch_size, device=self.method.device)).to(dtype=torch.bool)

        # must use reduction='sum' and divide by 2 * batch_size
        # credit to https://github.com/Yunfan-Li/Contrastive-Clustering/
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')

        self.contrastive_loss_weight = self.config.get('loss_weight', 1.0)
        self.running_losses['contrastive_loss'] = torch.zeros(1, requires_grad=False, device=self.method.device)
        self.loss_optimizer_groups['contrastive_loss'] = self.config.get('loss_optimizer_group_name', 'default')

        self.batch_augmentation_names = \
            [self.config['batch_augmentation_name_1'], self.config['batch_augmentation_name_2']]

    def iteration(self, epoch_num, epoch_iter_num, phase_iter_num, batch_samples, batch_labels):
        success, err_msg = super().iteration(epoch_num, epoch_iter_num, phase_iter_num, batch_samples, batch_labels)
        if not success:
            return success, err_msg

        # credits to https://github.com/Yunfan-Li/Contrastive-Clustering/

        batch_size = self.dataset_cfg['batch_size']

        # batch_samples will have two entries with augmentations 1 and 2 at dim 0
        batch_samples_aug_x1 = batch_samples[0]
        batch_samples_aug_x2 = batch_samples[1]

        contrastive_head_output_x1 = \
            F.normalize(self.contrastive_learning_head(batch_samples_aug_x1).view(batch_size, -1), dim=1)
        contrastive_head_output_x2 = \
            F.normalize(self.contrastive_learning_head(batch_samples_aug_x2).view(batch_size, -1), dim=1)

        cosine_similarities_x1_x1_is = contrastive_head_output_x1.repeat_interleave(batch_size, dim=0)
        cosine_similarities_x1_x1_js = contrastive_head_output_x1.repeat(batch_size, 1)

        cosine_similarities_x1_x1 = (self.cos_sim(cosine_similarities_x1_x1_is, cosine_similarities_x1_x1_js)
                                     / self.temperature_parameter).view(batch_size, -1)

        cosine_similarities_x2_x2_is = contrastive_head_output_x2.repeat_interleave(batch_size, dim=0)
        cosine_similarities_x2_x2_js = contrastive_head_output_x2.repeat(batch_size, 1)
        cosine_similarities_x2_x2 = (self.cos_sim(cosine_similarities_x2_x2_is, cosine_similarities_x2_x2_js)
                                     / self.temperature_parameter).view(batch_size, -1)

        cosine_similarities_x1_x2_is = cosine_similarities_x1_x1_is
        cosine_similarities_x1_x2_js = cosine_similarities_x2_x2_js
        cosine_similarities_x1_x2 = (self.cos_sim(cosine_similarities_x1_x2_is, cosine_similarities_x1_x2_js)
                                     / self.temperature_parameter).view(batch_size, -1)

        cosine_similarities_x1_x2_x1_x1 = \
            torch.cat((cosine_similarities_x1_x2,
                       cosine_similarities_x1_x1[self.diagonal_filter].view(batch_size, -1)), dim=1)
        cosine_similarities_x1_x2_x2_x2 = \
            torch.cat((cosine_similarities_x1_x2,
                       cosine_similarities_x2_x2[self.diagonal_filter].view(batch_size, -1)), dim=1)

        cosine_similarities = torch.cat((cosine_similarities_x1_x2_x1_x1, cosine_similarities_x1_x2_x2_x2), dim=0)

        labels = torch.arange(0, batch_size, 1, device=self.method.device, dtype=torch.long).repeat(2)
        contrastive_loss = self.contrastive_loss_weight * self.ce_loss(cosine_similarities, labels) / (2.0 * batch_size)

        self.report_losses({'contrastive_loss': contrastive_loss}, epoch_num, epoch_iter_num, phase_iter_num)
        return True, ''


class LearningInvarianceToTransformationsByUsingAssignmentStatisticsVectorsAndDataAugmentation(DesignPattern):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.initialization_success:
            return

        num_clusters = self.dataset_cfg['num_clusters']
        self.diagonal_filter = \
            (torch.ones((num_clusters, num_clusters), device=self.method.device) -
             torch.eye(num_clusters, device=self.method.device)).to(dtype=torch.bool)

        self.temperature_parameter = self.config.get('temperature_parameter', 1.0)
        self.cos_sim = nn.CosineSimilarity(dim=1)

        # must use reduction='sum' and divide by 2 * cluster count
        # credit to https://github.com/Yunfan-Li/Contrastive-Clustering/
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')
        self.contrastive_loss_weight = self.config.get('loss_weight', 1.0)
        self.running_losses['contrastive_loss'] = torch.zeros(1, requires_grad=False, device=self.method.device)
        self.loss_optimizer_groups['contrastive_loss'] = self.config.get('loss_optimizer_group_name', 'default')

        self.batch_augmentation_names = \
            [self.config['batch_augmentation_name_1'], self.config['batch_augmentation_name_2']]

    def iteration(self, epoch_num, epoch_iter_num, phase_iter_num, batch_samples, batch_labels):
        success, err_msg = super().iteration(epoch_num, epoch_iter_num, phase_iter_num, batch_samples, batch_labels)
        if not success:
            return success, err_msg

        # credits to https://github.com/Yunfan-Li/Contrastive-Clustering/

        num_clusters = self.dataset_cfg['num_clusters']

        # batch_samples will have two entries with augmentations 1 and 2 at dim 0
        batch_samples_aug_x1 = batch_samples[0]
        batch_samples_aug_x2 = batch_samples[1]

        batch_cluster_assignments_x1 = \
            self.get_cluster_assignments(batch_samples_aug_x1, epoch_num, epoch_iter_num, phase_iter_num, False)
        batch_cluster_assignments_x2 = \
            self.get_cluster_assignments(batch_samples_aug_x2, epoch_num, epoch_iter_num, phase_iter_num, False)
        batch_cluster_assignments_x1_t = batch_cluster_assignments_x1.transpose(0, 1)
        batch_cluster_assignments_x2_t = batch_cluster_assignments_x2.transpose(0, 1)

        cosine_similarities_x1_x1_is = batch_cluster_assignments_x1_t.repeat_interleave(num_clusters, dim=0)
        cosine_similarities_x1_x1_js = batch_cluster_assignments_x1_t.repeat(num_clusters, 1)
        cosine_similarities_x1_x1 = (self.cos_sim(cosine_similarities_x1_x1_is, cosine_similarities_x1_x1_js)
                                     / self.temperature_parameter).view(num_clusters, -1)

        cosine_similarities_x2_x2_is = batch_cluster_assignments_x2_t.repeat_interleave(num_clusters, dim=0)
        cosine_similarities_x2_x2_js = batch_cluster_assignments_x2_t.repeat(num_clusters, 1)
        cosine_similarities_x2_x2 = (self.cos_sim(cosine_similarities_x2_x2_is, cosine_similarities_x2_x2_js)
                                     / self.temperature_parameter).view(num_clusters, -1)

        cosine_similarities_x1_x2_is = cosine_similarities_x1_x1_is
        cosine_similarities_x1_x2_js = cosine_similarities_x2_x2_js
        cosine_similarities_x1_x2 = (self.cos_sim(cosine_similarities_x1_x2_is, cosine_similarities_x1_x2_js)
                                     / self.temperature_parameter).view(num_clusters, -1)

        cosine_similarities_x1_x2_x1_x1 = \
            torch.cat((cosine_similarities_x1_x2,
                       cosine_similarities_x1_x1[self.diagonal_filter].view(num_clusters, -1)), dim=1)
        cosine_similarities_x1_x2_x2_x2 = \
            torch.cat((cosine_similarities_x1_x2,
                       cosine_similarities_x2_x2[self.diagonal_filter].view(num_clusters, -1)), dim=1)

        cosine_similarities = torch.cat((cosine_similarities_x1_x2_x1_x1, cosine_similarities_x1_x2_x2_x2), dim=0)

        labels = torch.arange(0, num_clusters, 1, device=self.method.device, dtype=torch.long).repeat(2)
        contrastive_loss = \
            self.contrastive_loss_weight * self.ce_loss(cosine_similarities, labels) / (2.0 * num_clusters)

        self.report_losses({'contrastive_loss': contrastive_loss}, epoch_num, epoch_iter_num, phase_iter_num)
        return True, ''


class PreventingClusterDegeneracyByMaximizingEntropyOfSoftAssignments(DesignPattern):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.initialization_success:
            return

        self.entropy_loss_weight = self.config.get('loss_weight', 1.0)
        self.running_losses['entropy_loss'] = torch.zeros(1, requires_grad=False, device=self.method.device)

        self.batch_augmentation_names = [self.config['batch_augmentation_name']] \
            if 'batch_augmentation_name' in self.config else []

    def iteration(self, epoch_num, epoch_iter_num, phase_iter_num, batch_samples, batch_labels):
        success, err_msg = super().iteration(epoch_num, epoch_iter_num, phase_iter_num, batch_samples, batch_labels)
        if not success:
            return success, err_msg

        # NOTE: in CC, data augmentation is used with this pattern;
        # to keep the pattern more general, no data augmentation is used here
        current_batch_assignments = \
            self.get_cluster_assignments(batch_samples, epoch_num, epoch_iter_num, phase_iter_num, try_get_logits=False)
        batch_assignment_matrix_l1_norm = torch.norm(current_batch_assignments, p=1)
        # P(y_i^k) in CC paper:
        normed_batch_assignments = torch.sum(current_batch_assignments, dim=0) / batch_assignment_matrix_l1_norm
        entropy_loss = self.entropy_loss_weight * (
                torch.log(torch.tensor(self.dataset_cfg['num_clusters'])) +
                torch.sum(normed_batch_assignments * utils.safe_log(normed_batch_assignments), dim=0))

        self.report_losses({'entropy_loss': entropy_loss}, epoch_num, epoch_iter_num, phase_iter_num)
        self.loss_optimizer_groups['entropy_loss'] = self.config.get('loss_optimizer_group_name', 'default')
        return True, ''


class TrainingFeatureExtractorByUsingAdversarialInterpolation(DesignPattern):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.initialization_success:
            return

        self.encoder = next(filter(lambda c: c.name == self.config['encoder_name'],
                                   self.method.mappings))
        if self.encoder is None:
            self.initialization_success = False
            self.initialization_err_msg = 'Error: encoder "%s" for design pattern "%s" not found ' % \
                                          (self.config['encoder_name'], self.pattern_name)
            return

        self.decoder = next(filter(lambda c: c.name == self.config['decoder_name'],
                                   self.method.mappings), None)
        if self.decoder is None:
            self.initialization_success = False
            self.initialization_err_msg = 'Error: decoder "%s" for design pattern "%s" not found ' % \
                                          (self.config['decoder_name'], self.pattern_name)
            return

        self.critic = \
            None if 'critic_name' not in self.config \
            else next(filter(lambda c: c.name == self.config['critic_name'], self.method.mappings), None)
        self.discriminator = \
            None if 'discriminator_name' not in self.config \
            else next(filter(lambda c: c.name == self.config['discriminator_name'], self.method.mappings), None)
        if self.critic is None and self.discriminator is None:
            self.initialization_success = False
            self.initialization_err_msg = 'Error: neither critic nor discriminator found for design pattern "%s"' % \
                                          self.pattern_name
            return

        # note that in some cases (e.g. pretraining of discriminator), one loss will have no associated optimizers

        self.critic_regularizing_hyperparameter = self.config.get('critic_regularizing_hyperparameter', 0.2)

        self.ce_loss = nn.BCELoss()
        self.autoencoder_loss_weight = self.config.get('autoencoder_loss_weight', 1.0)
        self.critic_loss_weight = self.config.get('critic_loss_weight', 1.0)
        self.discriminator_loss_weight = self.config.get('discriminator_loss_weight', 1.0)
        self.running_losses['adversarial_mapping_loss'] = torch.zeros(1, requires_grad=False,
                                                                        device=self.method.device)
        self.running_losses['autoencoder_loss'] = torch.zeros(1, requires_grad=False, device=self.method.device)

        adversarial_mapping_optimizer_group_name_key = \
            ('discriminator' if self.critic is None else 'critic') + '_loss_optimizer_group_name'
        self.loss_optimizer_groups['adversarial_mapping_loss'] = \
            self.config[adversarial_mapping_optimizer_group_name_key]
        self.loss_optimizer_groups['autoencoder_loss'] = self.config['autoencoder_loss_optimizer_group_name']
        if self.loss_optimizer_groups['adversarial_mapping_loss'] == self.loss_optimizer_groups['autoencoder_loss']:
            self.method.log(('WARNING: using same optimizer group "%s" for "autoencoder_loss" and '
                             '"adversarial_mapping_loss" in "%s" design pattern; this is not recommended') %
                            (self.loss_optimizer_groups['adversarial_mapping_loss'], self.pattern_name))

        self.batch_augmentation_names = [self.config['batch_augmentation_name']] \
            if 'batch_augmentation_name' in self.config else []

    def iteration(self, epoch_num, epoch_iter_num, phase_iter_num, batch_samples, batch_labels):
        success, err_msg = super().iteration(epoch_num, epoch_iter_num, phase_iter_num, batch_samples, batch_labels)
        if not success:
            return success, err_msg

        batch_size = self.dataset_cfg['batch_size']

        adversarial_mapping_loss = None
        autoencoder_loss = None

        if self.critic is not None:
            half_batch_size = batch_size // 2
            inputs_x1 = batch_samples[:half_batch_size]
            inputs_x2 = batch_samples[half_batch_size:]

            features_x1 = self.encoder(inputs_x1)
            features_x2 = self.encoder(inputs_x2)
            reconstruction_x1 = self.decoder(features_x1)
            interpolated_reconstruction_of_x1 = \
                self.critic_regularizing_hyperparameter * inputs_x1 + \
                (1 - self.critic_regularizing_hyperparameter) * reconstruction_x1

            # divide by 2 to resolve ambiguity between alpha and 1-alpha
            # (see p. 3 of https://arxiv.org/abs/1807.07543v2)
            alpha = torch.rand(half_batch_size).to(self.method.device) / 2
            alpha_matrix = alpha.repeat_interleave(repeats=features_x1.shape[1], dim=0).view(half_batch_size, -1)
            interpolated_features = torch.mul(alpha_matrix, features_x1) + torch.mul(1 - alpha_matrix, features_x2)
            reconstruction_of_interpolated_features = self.decoder(interpolated_features)

            critic_regression_real_detached = self.critic(interpolated_reconstruction_of_x1.detach())
            critic_regression_interpolated_detached = self.critic(reconstruction_of_interpolated_features.detach())
            # no "sum" needed here, as the shape of critic_regression_*_detached is half_batch_size x 1
            adversarial_mapping_loss = \
                self.critic_loss_weight * (
                        (critic_regression_real_detached ** 2.0).mean() +
                        ((critic_regression_interpolated_detached -
                          alpha.view(half_batch_size, 1)) ** 2.0).mean())

            critic_regression_interpolated = self.critic(reconstruction_of_interpolated_features)
            autoencoder_loss = \
                self.autoencoder_loss_weight * (critic_regression_interpolated ** 2.0).mean()
        elif self.discriminator is not None:
            discriminator_assessment_real = self.discriminator(batch_samples)
            discriminator_assessment_generated_detached = \
                self.discriminator(self.decoder(self.encoder(batch_samples)).detach())
            labels_discriminator = torch.cat((torch.ones((batch_size,), device=self.method.device),
                                              torch.zeros((batch_size,), device=self.method.device)), dim=0)
            adversarial_mapping_loss = \
                self.discriminator_loss_weight * (
                    self.ce_loss(torch.cat((discriminator_assessment_real,
                                            discriminator_assessment_generated_detached), dim=0).view(-1),
                                 labels_discriminator))

            discriminator_assessment_generated = self.discriminator(self.decoder(self.encoder(batch_samples)))
            labels_autoencoder = torch.ones((batch_size,), device=self.method.device)
            autoencoder_loss = \
                self.autoencoder_loss_weight * \
                self.ce_loss(discriminator_assessment_generated.view(-1), labels_autoencoder)

        self.report_losses({'adversarial_mapping_loss': adversarial_mapping_loss,
                            'autoencoder_loss': autoencoder_loss}, epoch_num, epoch_iter_num, phase_iter_num)
        return True, ''


class FacilitatingTrainingOfFeatureExtractorByUsingLayerWisePretraining(DesignPattern):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.initialization_success:
            return

        self.encoder = next(filter(lambda c: c.name == self.config['encoder_name'], self.method.mappings), None)
        if self.encoder is None:
            self.initialization_success, self.initialization_err_msg = \
                False, 'Error: could not find encoder for design pattern "%s"' % self.pattern_name
            return

        self.decoder = next(filter(lambda c: c.name == self.config['decoder_name'], self.method.mappings), None)
        if self.decoder is None:
            self.initialization_success, self.initialization_err_msg = \
                False, 'Error: could not find decoder for design pattern "%s"' % self.pattern_name
            return

        self.dropout_rate = self.config.get('dropout_rate', 0.0)

        self.encoder_decoder_layer_pairs = []

        for encoder_decoder_layer_pairs_item_cfg in self.config['encoder_decoder_layer_pairs']:
            encoder_decoder_layer_pairs_item = \
                {'train_for_iterations': encoder_decoder_layer_pairs_item_cfg['train_for_iterations'],
                 'encoder_layers': [], 'decoder_layers': [], 'encoder_layer_names': [], 'decoder_layer_names': []}

            encoder_layer_names = encoder_decoder_layer_pairs_item_cfg['encoder_layer_names']
            for layer_name in encoder_layer_names:
                layer = self.encoder.layers.get(layer_name, None)
                if layer is None:
                    self.initialization_success, self.initialization_err_msg = \
                        False, 'Error: could not find layer "%s" for encoder of design pattern "%s"' % \
                        (layer_name, self.pattern_name)
                    return
                encoder_decoder_layer_pairs_item['encoder_layers'].append(layer)
                encoder_decoder_layer_pairs_item['encoder_layer_names'].append(layer_name)

            decoder_layer_names = encoder_decoder_layer_pairs_item_cfg['decoder_layer_names']
            for layer_name in decoder_layer_names:
                layer = self.decoder.layers.get(layer_name, None)
                if layer is None:
                    self.initialization_success, self.initialization_err_msg = \
                        False, 'Error: could not find layer "%s" for decoder of design pattern "%s"' % \
                        (layer_name, self.pattern_name)
                    return
                encoder_decoder_layer_pairs_item['decoder_layers'].append(layer)
                encoder_decoder_layer_pairs_item['decoder_layer_names'].append(layer_name)

            self.encoder_decoder_layer_pairs.append(encoder_decoder_layer_pairs_item)

        self.current_encoder_decoder_layer_pair_item_index = -1
        self.current_encoder_prior_network = None  # encoder network up to layers currently being trained
        self.current_encoder_layer_network = None  # layers of the encoder currently being trained
        self.current_decoder_layer_network = None  # layers of the decoder currently being trained

        self.reconstruction_loss_weight = self.config.get('loss_weight', 1.0)
        self.running_losses['reconstruction_loss'] = torch.zeros(1, requires_grad=False, device=self.method.device)
        self.loss_optimizer_groups['reconstruction_loss'] = self.config.get('loss_optimizer_group_name', 'default')

        self.batch_augmentation_names = [self.config['batch_augmentation_name']] \
            if 'batch_augmentation_name' in self.config else []

        self.finished = False

    def iteration(self, epoch_num, epoch_iter_num, phase_iter_num, batch_samples, batch_labels):
        success, err_msg = super().iteration(epoch_num, epoch_iter_num, phase_iter_num, batch_samples, batch_labels)
        if not success:
            return success, err_msg

        if self.finished:
            return True, ''

        # check if layer pairs changed; modify variables if needed

        current_iter = phase_iter_num - 1  # phase_iter_num starts at 1
        iter_cumulative_sum = 0
        found_current_range = False  # check if finished
        for index, encoder_decoder_layer_pairs_item in enumerate(self.encoder_decoder_layer_pairs, start=0):
            encoder_decoder_layer_pairs_item_num_iters = encoder_decoder_layer_pairs_item['train_for_iterations']
            if current_iter < iter_cumulative_sum + encoder_decoder_layer_pairs_item_num_iters:
                found_current_range = True
                if index > self.current_encoder_decoder_layer_pair_item_index:
                    # layer pairs changed; modify variables
                    self.current_encoder_decoder_layer_pair_item_index = index
                    prior_encoder_layers = []
                    for index_new in range(0, index):
                        prior_encoder_layers.extend([nn.Dropout(p=self.dropout_rate),
                                                     *self.encoder_decoder_layer_pairs[index_new]['encoder_layers']])

                    self.current_encoder_prior_network = nn.Sequential(*prior_encoder_layers)
                    self.current_encoder_layer_network = \
                        nn.Sequential(*encoder_decoder_layer_pairs_item['encoder_layers'])
                    self.current_decoder_layer_network = \
                        nn.Sequential(*encoder_decoder_layer_pairs_item['decoder_layers'])
                    self.method.log('Pattern "%s": now pretraining layers %s of encoder and %s of decoder' %
                                    (self.pattern_name,
                                     str(encoder_decoder_layer_pairs_item['encoder_layer_names']),
                                     str(encoder_decoder_layer_pairs_item['decoder_layer_names'])))
                break

            iter_cumulative_sum += encoder_decoder_layer_pairs_item_num_iters

        if not found_current_range:
            self.method.log('Pattern "%s": finished pretraining all layers' % self.pattern_name)
            self.finished = True
            return True, ''

        batch_size = batch_samples.shape[0]

        input = self.current_encoder_prior_network(batch_samples).detach()
        features = self.current_encoder_layer_network(input)
        reconstruction = self.current_decoder_layer_network(F.dropout(features, p=self.dropout_rate))
        reconstruction_loss = self.reconstruction_loss_weight * ((input - reconstruction) ** 2.0).sum() / batch_size
        self.report_losses({'reconstruction_loss': reconstruction_loss}, epoch_num, epoch_iter_num, phase_iter_num)
        return True, ''


class EncouragingClusterFormationByReinforcingCurrentAssignmentOfSamplesToClusters(DesignPattern):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.initialization_success:
            return

        self.num_clusters = self.dataset_cfg['num_clusters']

        self.sample_representation = self.config['sample_representation']
        self.cluster_representation = self.config['cluster_representation']
        self.associated_encoder = next(filter(lambda c: c.name == self.config['associated_encoder_name'],
                                              self.method.mappings), None) \
            if self.sample_representation == 'decoder_reconstructed_sample' else None
        self.associated_decoder = next(filter(lambda c: c.name == self.config['associated_decoder_name'],
                                              self.method.mappings), None) \
            if (self.sample_representation == 'decoder_reconstructed_sample'
                or self.cluster_representation == 'centroid_processed_by_decoder') else None
        if self.sample_representation == 'decoder_reconstructed_sample' and self.associated_encoder is None:
            self.initialization_err_msg = 'Error: design pattern "%s" could not find mapping "%s"' \
                                          % (self.pattern_name, self.config['associated_encoder_name'])
            return
        if (self.sample_representation == 'decoder_reconstructed_sample'
                or self.cluster_representation == 'centroid_processed_by_decoder') and self.associated_decoder is None:
            self.initialization_err_msg = 'Error: design pattern "%s" could not find mapping "%s"' \
                                          % (self.pattern_name, self.config['associated_decoder_name'])
            return

        self.associated_feature_extractor = self.config.get('associated_feature_extractor_name', next(
            filter(lambda c: c.type == 'feature_extractor', self.method.mappings), None))
        if self.sample_representation == 'feature_representation' and self.associated_feature_extractor is None:
            self.initialization_err_msg = 'Error: design pattern "%s" could not find associated feature extractor.' \
                                          % self.pattern_name

        self.sample_selection_criterion = self.config['sample_selection_criterion']
        self.cluster_selection_criterion = self.config['cluster_selection_criterion']

        self.pair_loss_weighting_term = self.config.get('sample_cluster_pair_loss_weighting_term', 'unit')
        self.threshold_on_feature_representation_distance_to_centroid = \
            self.config.get('threshold_on_feature_representation_distance_to_centroid', None)
        self.threshold_on_soft_assignment_to_cluster = \
            self.config.get('threshold_on_soft_assignment_to_cluster', None)

        self.loss_type = self.config['loss_type']
        self.similarity_loss_weight = self.config.get('loss_weight', 1.0)
        self.running_losses['similarity_loss'] = torch.zeros(1, requires_grad=False, device=self.method.device)
        self.loss_optimizer_groups['similarity_loss'] = self.config.get('loss_optimizer_group_name', 'default')

        self.batch_augmentation_names = [self.config['batch_augmentation_name']] \
            if 'batch_augmentation_name' in self.config else []

    def iteration(self, epoch_num, epoch_iter_num, phase_iter_num, batch_samples, batch_labels):
        success, err_msg = super().iteration(epoch_num, epoch_iter_num, phase_iter_num, batch_samples, batch_labels)
        if not success:
            return success, err_msg

        batch_size = batch_samples.shape[0]

        sample_representations = None
        # note: could still use feature representation as sample representation
        cluster_assignments = \
            self.get_cluster_assignments(batch_samples, epoch_num, epoch_iter_num, phase_iter_num, try_get_logits=False)
        if self.sample_representation == 'original_sample':
            sample_representations = batch_samples
        elif self.sample_representation == 'decoder_reconstructed_sample':
            sample_representations = self.associated_decoder(self.associated_encoder(batch_samples))
        elif self.sample_representation == 'feature_representation':
            sample_representations = self.associated_feature_extractor(batch_samples)
        elif self.sample_representation == 'soft_assignments':
            sample_representations = \
                self.get_cluster_assignments(batch_samples, epoch_num, epoch_iter_num, phase_iter_num,
                                             try_get_logits=True)

        cluster_representations = None
        if self.cluster_representation == 'centroid':
            cluster_representations = self.method.current_centroids
        elif self.cluster_representation == 'centroid_processed_by_decoder':
            cluster_representations = self.associated_decoder(self.method.current_centroids)
        elif self.cluster_representation == 'pseudolabel':
            # don't turn to LongTensor yet; could still use L2 loss
            cluster_representations = F.one_hot(torch.argmax(cluster_assignments, dim=1),
                                                num_classes=self.num_clusters).to(dtype=sample_representations.dtype)

        # compute distances between feature representations and centroids if necessary
        feature_representation_centroid_distances_squared = None

        if any([self.cluster_selection_criterion == 'nearest_centroid',
                self.sample_selection_criterion == 'threshold_on_feature_representation_distance_to_centroid',
                self.pair_loss_weighting_term == 'l2_distance_of_feature_representation_to_centroid',
                self.pair_loss_weighting_term ==
                'gaussian_kernel_based_similarity_of_feature_representation_to_centroid']):
            if self.method.current_centroids is None:
                return False, 'Error: design pattern "%s" cannot proceed as centroids not initialized' % \
                       self.pattern_name

            # if any of the conditions above met, must use feature representation as sample representation
            feature_representations_is = \
                sample_representations.unsqueeze(dim=1).repeat_interleave(repeats=self.num_clusters, dim=1)
            centroids_js = self.method.current_centroids.unsqueeze(dim=0).repeat(batch_size, 1, 1)
            feature_representation_centroid_distances_squared = \
                torch.sum((feature_representations_is - centroids_js) ** 2.0, dim=2)

        # for each sample, determine associated clusters
        sample_cluster_indices = None
        if self.cluster_selection_criterion == 'all_clusters':
            sample_cluster_indices = torch.ones((batch_size, self.num_clusters), device=self.method.device)
        elif self.cluster_selection_criterion == 'nearest_centroid':
            sample_cluster_indices = \
                F.one_hot(feature_representation_centroid_distances_squared.argmin(dim=1),
                          num_classes=self.num_clusters)\
                 .to(dtype=batch_samples.dtype)
        elif self.cluster_selection_criterion == 'argmax_of_soft_assignments':
            sample_cluster_indices = \
                F.one_hot(sample_representations.argmax(dim=1), num_classes=self.num_clusters)\
                 .to(dtype=batch_samples.dtype)

        # for each cluster, select associated samples
        if self.sample_selection_criterion == 'threshold_on_feature_representation_distance_to_centroid':
            # warning: extend *existing* filtering (cannot use 1.0 and 0.0)
            sample_cluster_indices = \
                torch.where(feature_representation_centroid_distances_squared <=
                            self.threshold_on_feature_representation_distance_to_centroid,
                            sample_cluster_indices, torch.zeros_like(sample_cluster_indices))
        elif self.sample_selection_criterion == 'threshold_on_soft_assignment_to_cluster':
            # warning: extend *existing* filtering (cannot use 1.0 and 0.0)
            sample_cluster_indices = \
                torch.where(cluster_assignments >= self.threshold_on_soft_assignment_to_cluster,
                            sample_cluster_indices, torch.zeros_like(sample_cluster_indices))

        sample_cluster_index_weights = None
        if self.pair_loss_weighting_term == 'soft_assignment_of_sample_to_cluster':
            sample_cluster_index_weights = torch.mul(cluster_assignments, sample_cluster_indices)
        elif self.pair_loss_weighting_term == 'l2_distance_of_feature_representation_to_centroid':
            sample_cluster_index_weights = torch.mul(feature_representation_centroid_distances_squared,
                                                     sample_cluster_indices)
        elif self.pair_loss_weighting_term == 'gaussian_kernel_based_similarity_of_feature_representation_to_centroid':
            sample_cluster_index_weights = F.softmax(-feature_representation_centroid_distances_squared, dim=1)
        else:
            sample_cluster_index_weights = torch.ones_like(sample_cluster_indices)

        nonzeros = sample_cluster_indices.nonzero()
        loss_sample_representations = sample_representations[nonzeros[:, 0]]
        loss_cluster_representations = cluster_representations[nonzeros[:, 1]]
        # select rows and columns in each row
        loss_weights = sample_cluster_index_weights[nonzeros[:, 0], nonzeros[:, 1]]
        similarity_loss = None
        if nonzeros.shape[0] > 0:
            if self.loss_type == 'l2':
                sample_cluster_distances_squared = \
                    torch.sum((loss_sample_representations - loss_cluster_representations) ** 2.0, dim=1)
                similarity_loss = \
                    self.similarity_loss_weight * \
                    (torch.sum(loss_weights * sample_cluster_distances_squared) / batch_size)
            elif self.loss_type == 'cross_entropy':
                similarity_loss = \
                    self.similarity_loss_weight * \
                    torch.sum((loss_weights * F.cross_entropy(loss_sample_representations,
                                                              torch.argmax(loss_cluster_representations, dim=1),
                                                              reduction='none'))) / batch_size
        else:
            similarity_loss = torch.zeros(1, device=self.method.device)

        self.report_losses({'similarity_loss': similarity_loss}, epoch_num, epoch_iter_num, phase_iter_num)
        return True, ''
