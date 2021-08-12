import design_pattern
import os
import itertools
import json
import math
import numpy as np
import random
import re
import sklearn.cluster
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import utils

from mapping import Mapping
from datetime import datetime
from schema import validate_method_schema


class Method:
    def __init__(self, config_file_path, log_file_path=None, log_timestamps=True):
        self.config_file_path = config_file_path
        self.config_dir_path = os.path.dirname(config_file_path)
        self.log_str = ''
        self.log_file_path = log_file_path
        self.name = None
        self.init_successful = False
        self.config = None
        self.config_str = ''
        self.mappings = []
        self.device = None
        self.log_write_pos = 0
        self.log_timestamps = log_timestamps
        self.current_centroids = None  # optimized by some design patterns
        # needed for highest_{last,average}_{acc,nmi,ari} classifier selection strategy:
        self.dataset_classifier_performances = dict()

        if not os.path.isfile(config_file_path):
            self.log('Method configuration file "%s" not found' % config_file_path)
        else:
            f = None
            try:
                with open(config_file_path, 'r') as f:
                    self.config_str = f.read()
            except (IOError, FileNotFoundError):
                self.log('Could not open or read method configuration file "%s"' % config_file_path)
            if f is not None:
                try:
                    self.config = json.loads(self.config_str)
                except json.JSONDecodeError:
                    pass
                if self.config is None:
                    self.log('Error parsing method configuration file "%s"' % config_file_path)
                else:
                    res, err_msg = validate_method_schema(self.config)
                    if res:
                        self.name = self.config.get('name', os.path.basename(config_file_path))
                        device_name = self.config.get('device', 'cuda_if_available')
                        if device_name == 'cuda_if_available':
                            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        else:
                            self.device = torch.device(device_name)

                        res, err_msg = self.init_mappings()
                        if res:
                            self.init_successful = True
                        else:
                            self.log('Error initializing mappings: ' + err_msg)
                    else:
                        self.log('Validation error: ' + err_msg)

        self.flush_log_to_file()

    def init_mappings(self):
        try:
            self.mappings.clear()  # reinitialization
            for mapping_cfg in self.config['mappings']:
                self.mappings.append(Mapping(mapping_cfg, self).to(self.device))
        except Exception as e:
            return False, str(e)
        return True, ''

    def log(self, msg, flush_to_file=False):
        line = (datetime.now().strftime('[%Y-%m-%d %H:%M:%S] ') if self.log_timestamps else '') + msg
        print(line)
        self.log_str += (os.linesep if self.log_str != '' else '') + line
        if flush_to_file:
            self.flush_log_to_file()
        return line

    def flush_log_to_file(self):
        if self.log_file_path is None:
            return False

        log_len = len(self.log_str)
        if log_len == self.log_write_pos:
            return True

        real_log_file_path = self.log_file_path % self.name
        try:
            log_file_exists = os.path.isfile(real_log_file_path)
            with open(real_log_file_path, 'a+') as f:
                f.write((os.linesep if log_file_exists else '') + self.log_str[self.log_write_pos:])
        except IOError as e:
            self.log('Error writing to log file "%s": %s' % (real_log_file_path, str(e)))
            return False

        self.log_write_pos = log_len
        return True

    def get_classification_function_from_classifier(self, classifier, try_get_logits):
        if classifier.type == 'feature_space_classifier':
            associated_feature_extractor = classifier.get_associated_feature_extractor()
            if associated_feature_extractor is None:
                self.log('Error: found no feature extractor to supply features to "%s"' % classifier.name)
                return None
            if try_get_logits:
                return lambda x: classifier.get_logits(associated_feature_extractor(x))
            else:
                return lambda x: classifier(associated_feature_extractor(x))
        elif classifier.type == 'sample_space_classifier':
            if try_get_logits:
                return lambda x: classifier.get_logits(x)
            else:
                return lambda x: classifier(x)

    def get_cluster_assignments(self, samples, num_clusters, epoch_num=-1, epoch_iter_num=-1, phase_iter_num=-1,
                                current_centroids=None, dataset_classifier_performances=None, try_get_logits=False):
        cluster_assignment_strategy_cfg = self.config['cluster_assignment_strategy']
        strategy_type = cluster_assignment_strategy_cfg['type']
        if strategy_type == 'classifier':
            classifier_selection_criterion = \
                cluster_assignment_strategy_cfg.get('classifier_selection_criterion', 'first_declared')
            is_performance_based_classifier_selection_criterion =\
                re.fullmatch(r'highest_(average|last)_(acc|nmi|ari)', classifier_selection_criterion) is not None
            performance_based_classifier_selection_criterion_performances_available = \
                is_performance_based_classifier_selection_criterion and len(dataset_classifier_performances) > 0\
                and len(dataset_classifier_performances[next(iter(dataset_classifier_performances))]) > 0
            classification_function = None

            if classifier_selection_criterion == 'first_declared':
                classifier = next(filter(lambda c: c.type.endswith('classifier'), self.mappings), None)
                classification_function = None if classifier is None\
                                          else self.get_classification_function_from_classifier(classifier,
                                                                                                try_get_logits)
            elif classifier_selection_criterion == 'name':
                classifier = next(filter(lambda c: c.name == cluster_assignment_strategy_cfg.get('classifier_name', ''),
                                         self.mappings), None)
                classification_function = None if classifier is None\
                                          else self.get_classification_function_from_classifier(classifier,
                                                                                                try_get_logits)
            elif is_performance_based_classifier_selection_criterion\
                    and performance_based_classifier_selection_criterion_performances_available:
                if dataset_classifier_performances is None:
                    return None

                perf_column_index = 0 if classifier_selection_criterion.endswith('_acc')\
                    else 1 if classifier_selection_criterion.endswith('_nmi')\
                    else 2
                classifier_aggregated_performances = []
                use_last_performance = '_last_' in classifier_selection_criterion
                for classifier in filter(lambda c: c.type.endswith('classifier'), self.mappings):
                    if classifier.name in dataset_classifier_performances:
                        classifier_performances = dataset_classifier_performances[classifier.name]
                        classifier_aggregated_performance =\
                            classifier_performances[-1][perf_column_index] if use_last_performance\
                            else sum(perf[perf_column_index] for perf in classifier_performances) /\
                            len(classifier_performances)

                        classifier_aggregated_performances.append([classifier, classifier_aggregated_performance])
                best_classifier = sorted(classifier_aggregated_performances, key=lambda pair: -pair[1])[0][0]
                classification_function = self.get_classification_function_from_classifier(best_classifier,
                                                                                           try_get_logits)
            elif classifier_selection_criterion == 'average'\
                    or (is_performance_based_classifier_selection_criterion
                        and not performance_based_classifier_selection_criterion_performances_available):
                classification_function = lambda x:\
                    torch.mean(torch.stack([self.get_classification_function_from_classifier(c, try_get_logits)(x)
                                            for c in self.mappings if c.type.endswith('classifier')]), dim=0)

            return None if classification_function is None else classification_function(samples)
        elif strategy_type == 'classical_clustering_in_feature_space_after_phase':
            associated_feature_extractor =\
                cluster_assignment_strategy_cfg.get('associated_feature_extractor_name', next(
                    filter(lambda c: c.type == 'feature_extractor', self.mappings), None))
            if associated_feature_extractor is not None:
                centroids, labels = self.perform_classical_clustering\
                    (cluster_assignment_strategy_cfg['classical_clustering_method_after_phase'],
                     associated_feature_extractor(samples),
                     num_clusters,
                     cluster_assignment_strategy_cfg.get('classical_clustering_after_phase_num_restarts', 20))
                return None if labels is None else F.one_hot(labels, num_classes=num_clusters).to(dtype=samples.dtype)
            return None
        elif strategy_type == 'feature_representation_centroid_similarity':
            if current_centroids is None:
                return None
            associated_feature_extractor =\
                cluster_assignment_strategy_cfg.get('associated_feature_extractor_name', next(
                    filter(lambda c: c.type == 'feature_extractor', self.mappings), None))
            if associated_feature_extractor is None:
                return None
            features = associated_feature_extractor(samples)
            num_samples = len(samples)

            similarity_measure = cluster_assignment_strategy_cfg['similarity_measure']
            if similarity_measure == 'student_t':
                student_t_alpha = 1.0
                features_is = features.repeat_interleave(num_clusters, dim=0)
                centroids_js = current_centroids.repeat(num_samples, 1)
                cluster_centroid_distances = torch.sum((features_is - centroids_js) ** 2.0, dim=1)
                cluster_assignments_unnormed = ((1.0 + ((cluster_centroid_distances ** 2.0) / student_t_alpha))
                                                ** (-(student_t_alpha + 1.0) / 2.0)).view(num_samples, -1)
                if try_get_logits:
                    return cluster_assignments_unnormed

                cluster_assignments_norm = torch.sum(cluster_assignments_unnormed, dim=1)\
                                                .repeat_interleave(num_clusters, dim=0).view(num_samples, -1)
                return cluster_assignments_unnormed / cluster_assignments_norm
            elif similarity_measure == 'gaussian':
                # determine current inverse temperature parameter
                inverse_temperature_cfg =\
                    cluster_assignment_strategy_cfg['gaussian_similarity_measure_inverse_temperature']
                alpha = None
                if 'fixed_value' in inverse_temperature_cfg:
                    alpha = inverse_temperature_cfg['fixed_value']
                else:
                    initial_alpha = inverse_temperature_cfg['annealing_initial_value']
                    final_alpha = inverse_temperature_cfg['annealing_final_value']
                    if phase_iter_num == -1:
                        alpha = final_alpha
                    else:
                        alpha = initial_alpha
                        iterations_per_alpha = inverse_temperature_cfg['annealing_iterations_per_value']
                        alpha_steps = 1 + phase_iter_num // iterations_per_alpha
                        for step in range(1, alpha_steps):  # see p. 6 of https://arxiv.org/abs/1806.10069
                            alpha *= 2.0 ** (1 / (math.log(step + 1) ** 2.0))
                        if alpha > final_alpha:
                            alpha = final_alpha

                features_is = features.repeat_interleave(num_clusters, dim=0)
                centroids_js = current_centroids.repeat(num_samples, 1)
                cluster_centroid_distances_squared = torch.sum((features_is - centroids_js) ** 2.0, dim=1)

                if try_get_logits:
                    return -alpha * cluster_centroid_distances_squared.view(num_samples, -1)
                else:
                    return F.softmax(-alpha * cluster_centroid_distances_squared.view(num_samples, -1), dim=1)

        return None

    def perform_classical_clustering(self, classical_clustering_method, sample_representations, num_clusters,
                                     num_restarts):
        if classical_clustering_method == 'k_means':
            representations_np = sample_representations.cpu().detach().numpy()
            k_means_result = \
                sklearn.cluster.KMeans(n_clusters=num_clusters, n_init=num_restarts).fit(representations_np)
            return \
                torch.from_numpy(k_means_result.cluster_centers_).to(self.device),\
                torch.from_numpy(k_means_result.labels_).to(dtype=torch.int64, device=self.device)
        return None, None

    def process(self):
        if not self.init_successful:
            return False, self.log('Cannot process method "%s" as initialization was not successful' % self.name,
                                   flush_to_file=True)

        self.log('Started processing method "%s"' % self.name)

        # only relevant for methods which use centroids (may e.g. be used in parallel to a classifier):
        # centroids can be kept across datasets, if 'reinitialize_mappings' is set to false
        self.current_centroids = None

        datasets = self.config['datasets']
        datasets.sort(key=lambda d: d.get('order', 0))
        phases = self.config['phases']
        phases.sort(key=lambda p: p['order'])
        for dataset_num, dataset_cfg in enumerate(datasets):
            dataset_name = dataset_cfg.get('name', dataset_cfg['dataset'])
            dataset_id = dataset_cfg['dataset']
            dataset_transforms_cfg = dataset_cfg.get('transforms', [])
            dataset_batch_size = dataset_cfg['batch_size']
            dataset_num_clusters = dataset_cfg['num_clusters']
            dataset_shuffle = dataset_cfg.get('shuffle', False)
            dataset_num_workers = dataset_cfg.get('num_workers', 0)
            dataset_pin_memory = dataset_cfg.get('pin_memory', False)
            dataset_reinitialize_mappings = dataset_cfg.get('reinitialize_mappings', True)
            dataset_index_of_first_sample_to_use = dataset_cfg.get('index_of_first_sample_to_use', -1)
            dataset_index_of_last_sample_to_use = dataset_cfg.get('index_of_last_sample_to_use', -1)

            if not dataset_cfg.get('enabled', True):
                self.log('Skipping dataset "%s" (disabled)' % dataset_name)

            self.log(('Processing dataset "%s"' % dataset_name) +
                     ('' if dataset_reinitialize_mappings or dataset_num == 0
                      else ' (retaining mapping parameters)'))

            if dataset_reinitialize_mappings:
                self.current_centroids = None  # reinitialize centroids
                res, err_msg = self.init_mappings()
                if not res:
                    return res, self.log('Error (re)initializing mappings for dataset "%s": %s' %
                                         (dataset_name, err_msg), flush_to_file=True)
                if dataset_num > 0:
                    self.log('Mapping parameters reinitialized')

            dataset_transforms = []
            for transform_cfg in dataset_transforms_cfg:
                dataset_transforms.append(utils.recursive_type_init(base_namespace=torchvision.transforms,
                                                                    config=transform_cfg,
                                                                    additional_keys_to_remove=[]))

            dataset_args = dict(dataset_cfg)
            dataset_cfg_keys_to_remove =\
                ['name', 'dataset', 'phases', 'order', 'enabled', 'reinitialize_mappings', 'transforms', 'batch_size',
                 'num_clusters', 'shuffle', 'num_workers', 'pin_memory', 'index_of_first_sample_to_use',
                 'index_of_last_sample_to_use', 'batch_augmentations']
            dataset_args['type'] = dataset_id
            dataset_args['transform'] = transforms.Compose(dataset_transforms) if len(dataset_transforms) > 0\
                else transforms.ToTensor()
            if 'root' in dataset_args:
                dataset_args['root'] = utils.resolve_relative_path(dataset_args['root'], self.config_dir_path)
            dataset = utils.recursive_type_init(base_namespace=torchvision.datasets, config=dataset_args,
                                                additional_keys_to_remove=dataset_cfg_keys_to_remove)
            subset = torch.utils.data.Subset(dataset,
                                             range(dataset_index_of_first_sample_to_use,
                                                   dataset_index_of_last_sample_to_use+1))\
                if dataset_index_of_first_sample_to_use > -1 else None

            # make sure we get batches in the same orders for all DataLoaders (needed to make batch augmentations match)
            dataset_loader_rng_seed = random.randint(0, 1e9)

            dataset_batch_augmentations_cfgs = dataset_cfg.get('batch_augmentations', [])

            def construct_dataset_loader(batch_size, force_no_shuffle=False, custom_dataset=None):
                dataset_loader_generator = torch.Generator()
                dataset_loader_generator.manual_seed(dataset_loader_rng_seed)
                return torch.utils.data.DataLoader(custom_dataset if custom_dataset is not None
                                                   else subset if dataset_index_of_first_sample_to_use > -1
                                                   else dataset,
                                                   batch_size=batch_size,
                                                   shuffle=dataset_shuffle and not force_no_shuffle,
                                                   num_workers=dataset_num_workers //
                                                   (1 + len(dataset_batch_augmentations_cfgs)),
                                                   generator=dataset_loader_generator,
                                                   pin_memory=dataset_pin_memory,
                                                   drop_last=True)  # drop non-full batches for simplicity

            dataset_loader = construct_dataset_loader(dataset_batch_size)
            dataset_num_samples = len(dataset_loader) * dataset_batch_size

            # construct datasets and loaders for each batch augmentation

            dataset_batch_augmentation_transforms = {}
            dataset_batch_augmentation_transform_datasets = {}
            dataset_batch_augmentation_transform_loaders = {}
            # note: iterators must be reinitialized every epoch due to the way iterators work in Python
            dataset_batch_augmentation_transform_iterators = {}

            for dataset_batch_augmentations_cfg in dataset_batch_augmentations_cfgs:
                batch_augmentation_name = dataset_batch_augmentations_cfg['name']
                batch_augmentation_transforms_cfg = dataset_batch_augmentations_cfg['transforms']
                batch_augmentation_transforms = []
                for transform_cfg in batch_augmentation_transforms_cfg:
                    batch_augmentation_transforms.append(
                        utils.recursive_type_init(base_namespace=torchvision.transforms,
                                                  config=transform_cfg,
                                                  additional_keys_to_remove=[]))

                batch_augmentation_transform = transforms.Compose(batch_augmentation_transforms)
                dataset_batch_augmentation_transforms[batch_augmentation_name] = batch_augmentation_transform

                dataset_args['transform'] = batch_augmentation_transform
                batch_augmentation_dataset =\
                    utils.recursive_type_init(base_namespace=torchvision.datasets,
                                              config=dataset_args,
                                              additional_keys_to_remove=dataset_cfg_keys_to_remove)
                dataset_batch_augmentation_transform_datasets[batch_augmentation_name] = batch_augmentation_dataset

                batch_augmentation_dataset_subset = \
                    torch.utils.data.Subset(batch_augmentation_dataset,
                                            range(dataset_index_of_first_sample_to_use,
                                                  dataset_index_of_last_sample_to_use + 1)) \
                        if dataset_index_of_first_sample_to_use > -1 else batch_augmentation_dataset

                batch_augmentation_loader = construct_dataset_loader(dataset_batch_size,
                                                                     custom_dataset=batch_augmentation_dataset_subset)
                dataset_batch_augmentation_transform_loaders[batch_augmentation_name] = batch_augmentation_loader

            # only relevant for methods which use centroids (may e.g. be used in parallel to a classifier):
            current_sample_centroid_assignments = None
            cluster_assignment_strategy_cfg = self.config['cluster_assignment_strategy']
            centroid_initialization_strategy =\
                cluster_assignment_strategy_cfg.get('centroid_initialization_strategy', None)
            centroid_recalculation_strategy =\
                cluster_assignment_strategy_cfg.get('centroid_recalculation_strategy', None)
            centroid_recalculation_interval =\
                self.config['cluster_assignment_strategy'].get('centroid_recalculation_interval', -1)
            use_centroids_during_phases = cluster_assignment_strategy_cfg.get('use_centroids_during_phases', [])

            # only relevant for methods which use soft assignments:
            current_soft_sample_cluster_assignments = None

            phases_to_use_with_dataset_cfg = list(filter(lambda p: p['name'] in dataset_cfg['phases'], phases)) \
                if 'phases' in dataset_cfg else phases

            # format: {'classifier_name': [[acc_1, nmi_1, ari_1], [acc_2, nmi_2, ari_2], ...]}
            # (statistics for each prior performance evaluation)
            for classifier in filter(lambda c: c.type.endswith('classifier'), self.mappings):
                if classifier.name in self.dataset_classifier_performances:
                    return False, self.log('Error: cannot use multiple classifiers with non-unique names',
                                           flush_to_file=True)
                self.dataset_classifier_performances[classifier.name] = []
            num_classifiers = len(self.dataset_classifier_performances)

            for phase_cfg in phases_to_use_with_dataset_cfg:
                phase_name = phase_cfg['name']
                phase_order = phase_cfg['order']
                phase_exit_criteria_cfg = phase_cfg['exit_criteria']
                phase_design_patterns_cfg = phase_cfg.get('design_patterns', [])
                phase_design_patterns = []
                phase_performance_evaluation_interval = phase_cfg.get('performance_evaluation_interval', -1)
                phase_needs_mapping_parameter_saving = 'save_mapping_parameters' in phase_cfg
                phase_has_max_total_iters = 'iterations' in phase_exit_criteria_cfg
                phase_max_total_iters = phase_exit_criteria_cfg.get('iterations', -1)

                if not phase_cfg.get('enabled', True):
                    self.log('Skipping phase "%s" (#%i; disabled)' % (phase_name, phase_order), flush_to_file=True)
                    continue

                self.log('Entered phase "%s" (#%i)' % (phase_name, phase_order), flush_to_file=True)

                # these values must be reinitialized during each phase as phase exit criteria may depend on them
                last_performance_evaluation_acc = 0.0
                last_performance_evaluation_nmi = 0.0
                last_performance_evaluation_ari = 0.0
                last_performance_evaluation_cluster_reassignment_ratio = 1.0
                last_performance_evaluation_one_hot_cluster_assignments = None
                iterations_since_centroid_initialization = 0
                iterations_since_last_performance_evaluation = 0

                # modifies last_performance_evaluation_*, iterations_since_last_performance_evaluation
                # returns success, err_msg
                def evaluate_performance_on_dataset(epoch_num, epoch_iter_num, phase_iter_num):
                    nonlocal last_performance_evaluation_acc, last_performance_evaluation_nmi, \
                        last_performance_evaluation_ari, last_performance_evaluation_cluster_reassignment_ratio, \
                        last_performance_evaluation_one_hot_cluster_assignments,\
                        iterations_since_last_performance_evaluation

                    iterations_since_last_performance_evaluation = 0

                    one_hot_cluster_assignments = None
                    all_sample_labels_one_hot = None
                    classifier_outputs_one_hot =\
                        {classifier.name: torch.empty((dataset_num_samples, dataset_num_clusters), device=self.device)
                         for classifier in filter(lambda c: c.type.endswith('classifier'), self.mappings)}\
                        if num_classifiers > 1 else None

                    cluster_assignment_strategy_type = self.config['cluster_assignment_strategy']['type']
                    # if possible, get cluster assignments and labels batch-wise, to alleviate memory problems
                    one_hot_cluster_assignments =\
                        torch.empty((dataset_num_samples, dataset_num_clusters), device=self.device)
                    all_sample_labels_one_hot =\
                        torch.empty((dataset_num_samples, dataset_num_clusters), device=self.device)

                    sample_loader_batch_size = dataset_num_samples\
                        if cluster_assignment_strategy_type == 'classical_clustering_in_feature_space_after_phase'\
                        else dataset_batch_size
                    sample_loader = construct_dataset_loader(batch_size=sample_loader_batch_size, force_no_shuffle=True)
                    for iter_num, iter_data in enumerate(sample_loader, start=0):  # note different indexing here
                        batch_samples, batch_labels = iter_data
                        batch_samples, batch_labels = batch_samples.to(self.device), batch_labels.to(self.device)
                        if len(batch_samples) != sample_loader_batch_size:
                            continue
                        index_range = range(iter_num*sample_loader_batch_size, (iter_num+1)*sample_loader_batch_size)
                        all_sample_labels_one_hot[index_range] = \
                            F.one_hot(batch_labels, num_classes=dataset_num_clusters).to(dtype=batch_samples.dtype)
                        current_batch_cluster_assignments =\
                            self.get_cluster_assignments(batch_samples,
                                                         dataset_num_clusters,
                                                         epoch_num, epoch_iter_num, phase_iter_num,
                                                         self.current_centroids,
                                                         self.dataset_classifier_performances,
                                                         try_get_logits=False)

                        # turn soft assignments to hard assignments:
                        one_hot_cluster_assignments[index_range] =\
                            F.one_hot(torch.argmax(current_batch_cluster_assignments, dim=1),
                                      num_classes=dataset_num_clusters).to(dtype=batch_samples.dtype)\
                            if cluster_assignment_strategy_type != 'classical_clustering_in_feature_space_after_phase'\
                            else current_batch_cluster_assignments

                        if num_classifiers > 1:
                            for classifier in filter(lambda c: c.type.endswith('classifier'), self.mappings):
                                classifier_output =\
                                    self.get_classification_function_from_classifier(classifier)(batch_samples)
                                classifier_outputs_one_hot[classifier.name][index_range] =\
                                    F.one_hot(torch.argmax(classifier_output, dim=1),
                                              num_classes=dataset_num_clusters).to(dtype=batch_samples.dtype)

                    if one_hot_cluster_assignments is None:
                        return False, 'Error: could not evaluate performance on dataset "%s"' % dataset_name

                    if last_performance_evaluation_one_hot_cluster_assignments is not None:
                        # note that if ... is None (no prior performance evaluations on this dataset),
                        # last_performance_evaluation_cluster_reassignment_ratio must remain 1.0,
                        # or else the min reassignment threshold may be incorrectly passed

                        last_performance_evaluation_cluster_reassignment_ratio =\
                            1.0 - torch.sum(torch.mul(last_performance_evaluation_one_hot_cluster_assignments,
                                                      one_hot_cluster_assignments)).cpu().item() / dataset_num_samples

                    last_performance_evaluation_one_hot_cluster_assignments = one_hot_cluster_assignments
                    last_performance_evaluation_acc, last_performance_evaluation_nmi, last_performance_evaluation_ari =\
                        utils.evaluate_clustering_performance(one_hot_cluster_assignments, all_sample_labels_one_hot)

                    # evaluate performance of each classifier if there are multiple ones
                    if len(self.dataset_classifier_performances) > 1:
                        for classifier in filter(lambda c: c.type.endswith('classifier'), self.mappings):
                            acc, nmi, ari =\
                                utils.evaluate_clustering_performance(classifier_outputs_one_hot[classifier.name],
                                                                      all_sample_labels_one_hot)
                            self.dataset_classifier_performances[classifier.name].append([acc, nmi, ari])
                            self.log(utils.get_performance_string(epoch_num, epoch_iter_num,
                                                                  acc, nmi, ari,
                                                                  'of classifier "%s"' % classifier.name))
                    return True, ''

                # returns: success, err_msg
                def update_centroids_by_classical_clustering(classical_clustering_method, num_restarts):
                    nonlocal current_sample_centroid_assignments

                    associated_feature_extractor =\
                        cluster_assignment_strategy_cfg.get('associated_feature_extractor_name',
                                                            next(filter(lambda c: c.type == 'feature_extractor',
                                                                        self.mappings), None))
                    if associated_feature_extractor is None:
                        return False, 'Error: no feature extractor found to initialize centroids'
                    sample_loader = construct_dataset_loader(dataset_num_samples)
                    all_samples, _ = next(iter(sample_loader))
                    all_samples = all_samples.to(self.device)
                    all_features = associated_feature_extractor(all_samples)

                    centroids, sample_centroid_assignments = self.perform_classical_clustering(
                        classical_clustering_method=classical_clustering_method,
                        sample_representations=all_features,
                        num_clusters=dataset_num_clusters,
                        num_restarts=num_restarts)
                    if centroids is None:
                        return False, 'Error: could not perform classical clustering to initialize centroids'

                    self.current_centroids = centroids
                    current_sample_centroid_assignments = sample_centroid_assignments
                    return True, ''

                # returns: success, err_msg
                def check_save_mapping_parameters(force_save, real_epoch_num=-1, real_epoch_iter_num=-1,
                                                    phase_iter_num=-1):
                    mappings_to_save_cfgs = phase_cfg['save_mapping_parameters'] if force_save\
                                              else filter(lambda mapping_cfg:
                                                          phase_iter_num % mapping_cfg['saving_interval'] == 0,
                                                          phase_cfg['save_mapping_parameters'])

                    if 'save_centroids' in phase_cfg:
                        mappings_to_save_cfgs = [*mappings_to_save_cfgs, phase_cfg['save_centroids']]

                    for save_mapping_parameters_cfg in mappings_to_save_cfgs:
                        is_centroids = 'mapping_name' not in save_mapping_parameters_cfg
                        mapping_name = 'centroids' if is_centroids\
                                         else save_mapping_parameters_cfg['mapping_name']
                        path_to_file_or_dir =\
                            utils.resolve_relative_path(save_mapping_parameters_cfg['path_to_file_or_dir'],
                                                        self.config_dir_path)

                        basename = os.path.basename(path_to_file_or_dir)
                        filename_supplied = '.' in basename

                        dir_path = os.path.dirname(path_to_file_or_dir) if filename_supplied\
                                          else path_to_file_or_dir
                        if not os.path.isdir(dir_path):
                            try:
                                os.makedirs(dir_path)
                            except Exception as e:
                                return False, ('Error: could not save parameters of mapping "%s": ' +
                                               'could not create directory "%s"') %\
                                              (mapping_name, dir_path)
                        keep_old_files = save_mapping_parameters_cfg.get('keep_old_files', False)
                        save_path =\
                            path_to_file_or_dir if filename_supplied\
                            else os.path.join(dir_path, utils.sanitize_filename(
                                                          '%s_%s_%s.pth' % (self.name, dataset_name, mapping_name)))

                        if keep_old_files:
                            orig_fn = save_path
                            counter = 1
                            last_mapping_save_path = save_mapping_parameters_cfg.get('last_save_path', None)
                            while os.path.isfile(save_path) and not save_path == last_mapping_save_path:
                                parts = os.path.basename(orig_fn).split('.')
                                save_path = os.path.join(dir_path, '.'.join(parts[:-1]) + ('_%i.' % counter) +
                                                         parts[-1])
                                counter += 1
                            save_mapping_parameters_cfg['last_save_path'] = save_path

                        if is_centroids:
                            if self.current_centroids is None:
                                return False, ('Error: could not save parameters of centroids ' +
                                               'into file "%s": centroids not initialized') % save_path
                            try:
                                torch.save({'centroids': self.current_centroids}, save_path)
                            except Exception as e:
                                return False, ('Error: could not save parameters of centroids ' +
                                               'into file "%s": %s') % (save_path, str(e))
                        else:
                            mapping = next(filter(lambda c: c.name == mapping_name, self.mappings), None)
                            if mapping is None:
                                return False, ('Error: could not save parameters of mapping "%s" ' +
                                               'into file "%s": mapping does not exist') % (mapping_name, save_path)

                            try:
                                torch.save(mapping.state_dict(), save_path)
                            except Exception as e:
                                return False, ('Error: could not save parameters of mapping "%s" ' +
                                               'into file "%s": %s') % (mapping_name, save_path, str(e))

                    return True, ''

                # returns: success, err_msg, do_exit_phase, exit_reason_msg
                def check_phase_exit_criteria(epoch_num, epoch_iter_num, real_epoch_num, real_epoch_iter_num,
                                              phase_iter_num):
                    nonlocal current_sample_centroid_assignments, phase_design_patterns

                    exit_criteria_check =\
                        [(epoch_num > phase_exit_criteria_cfg.get('epochs', epoch_num),
                         'max epochs reached'),
                         (phase_has_max_total_iters and phase_iter_num > phase_max_total_iters,
                          'max iterations reached'),
                         ('performance_evaluation_acc_gt' in phase_exit_criteria_cfg and
                          last_performance_evaluation_acc > phase_exit_criteria_cfg['performance_evaluation_acc_gt'],
                          'max ACC performance threshold passed'),
                         ('performance_evaluation_nmi' in phase_exit_criteria_cfg and
                          last_performance_evaluation_nmi > phase_exit_criteria_cfg['performance_evaluation_nmi_gt'],
                          'max NMI performance threshold passed'),
                         ('performance_evaluation_ari_gt' in phase_exit_criteria_cfg and
                          last_performance_evaluation_ari > phase_exit_criteria_cfg['performance_evaluation_ari_gt'],
                          'max ARI performance threshold passed'),
                         ('performance_evaluation_cluster_reassignment_ratio_lt' in phase_exit_criteria_cfg and
                          last_performance_evaluation_cluster_reassignment_ratio <
                          phase_exit_criteria_cfg['performance_evaluation_cluster_reassignment_ratio_lt'],
                          'min cluster reassignment ratio threshold passed'),
                         ('performance_evaluation_acc_lt_after_iterations' in phase_exit_criteria_cfg and
                          phase_exit_criteria_cfg['performance_evaluation_acc_lt_after_iterations']['iterations']
                          <= phase_iter_num and
                          phase_exit_criteria_cfg['performance_evaluation_acc_lt_after_iterations']['value'] >
                          last_performance_evaluation_acc, 'min ACC performance threshold after specified number of ' +
                                                           'iterations in phase passed'),
                         ('performance_evaluation_nmi_lt_after_iterations' in phase_exit_criteria_cfg and
                          phase_exit_criteria_cfg['performance_evaluation_nmi_lt_after_iterations']['iterations']
                          <= phase_iter_num and
                          phase_exit_criteria_cfg['performance_evaluation_nmi_lt_after_iterations']['value'] >
                          last_performance_evaluation_nmi, 'min NMI performance threshold after specified number of ' +
                                                           'iterations in phase passed'),
                         ('performance_evaluation_ari_lt_after_iterations' in phase_exit_criteria_cfg and
                          phase_exit_criteria_cfg['performance_evaluation_ari_lt_after_iterations']['iterations']
                          <= phase_iter_num and
                          phase_exit_criteria_cfg['performance_evaluation_ari_lt_after_iterations']['value'] >
                          last_performance_evaluation_ari, 'min ARI performance threshold after specified number of ' +
                                                           'iterations in phase passed')]
                    fulfilled_phase_exit_criteria = map(lambda c: c[1], filter(lambda c: c[0], exit_criteria_check))

                    exit_reason_msg = next(fulfilled_phase_exit_criteria, None)
                    if exit_reason_msg is not None:
                        for design_pattern in phase_design_patterns:
                            success, err_msg = design_pattern.exit_phase()
                            if not success:
                                return success, err_msg, True, exit_reason_msg

                        if cluster_assignment_strategy_cfg['type'] ==\
                                'classical_clustering_in_feature_space_after_phase':
                            if cluster_assignment_strategy_cfg['classical_clustering_after_phase'] ==\
                                    phase_name:
                                # performance will be reported after exiting phase
                                success, err_msg = evaluate_performance_on_dataset(real_epoch_num,
                                                                                   real_epoch_iter_num,
                                                                                   phase_iter_num)
                                if not success:
                                    return success, self.log(err_msg, flush_to_file=True)

                        if phase_needs_mapping_parameter_saving:
                            success, err_msg = check_save_mapping_parameters\
                                (True, real_epoch_num, real_epoch_iter_num, phase_iter_num)
                            if not success:
                                return success, err_msg, True, exit_reason_msg

                        self.log('Exiting phase "%s": %s' % (phase_name, exit_reason_msg), flush_to_file=True)

                        return True, '', True, exit_reason_msg
                    else:
                        return True, '', False, ''

                if 'load_mapping_parameters' in phase_cfg:
                    for load_mapping_parameters_cfg in phase_cfg['load_mapping_parameters']:
                        mapping_name = load_mapping_parameters_cfg['mapping_name']
                        path_to_file = utils.resolve_relative_path(load_mapping_parameters_cfg['path_to_file'],
                                                                   self.config_dir_path)
                        if not os.path.isfile(path_to_file):
                            return False, self.log(('Error: could not initialize parameters of mapping "%s" ' +
                                                    'from file "%s": file does not exist') %
                                                   (mapping_name, path_to_file), flush_to_file=True)
                        mapping = next(filter(lambda c: c.name == mapping_name, self.mappings), None)
                        if mapping is None:
                            return False, self.log(('Error: could not initialize parameters of mapping "%s" ' +
                                                    'from file "%s": mapping does not exist') %
                                                   (mapping_name, path_to_file), flush_to_file=True)

                        try:
                            mapping.load_state_dict(torch.load(path_to_file, map_location=self.device))
                        except Exception as e:
                            return False, self.log(('Error: could not initialize parameters of mapping "%s" ' +
                                                    'from file "%s": %s') %
                                                   (mapping_name, path_to_file, str(e)), flush_to_file=True)
                        self.log('Parameters for mapping "%s" loaded from file "%s"' % (mapping_name, path_to_file))

                if self.current_centroids is None and phase_name in use_centroids_during_phases:
                    associated_feature_extractor = \
                        cluster_assignment_strategy_cfg.get('associated_feature_extractor_name',
                                                            next(filter(lambda c: c.type == 'feature_extractor',
                                                                        self.mappings), None))
                    if associated_feature_extractor is None:
                        return False, 'Error: no feature extractor found to initialize centroids'

                    if centroid_initialization_strategy == 'classical_clustering':
                        success, err_msg =\
                            update_centroids_by_classical_clustering(
                                cluster_assignment_strategy_cfg['centroid_initialization_classical_clustering_method'],
                                cluster_assignment_strategy_cfg.
                                    get('centroid_initialization_classical_clustering_num_restarts', 20))
                        if not success:
                            return success, self.log(err_msg, flush_to_file=True)
                    elif centroid_initialization_strategy == 'random_feature_representations':
                        random_indices = random.sample(dataset_num_samples if dataset_index_of_first_sample_to_use < 0
                                                       else range(dataset_index_of_first_sample_to_use,
                                                                  dataset_index_of_last_sample_to_use+1),
                                                       dataset_num_clusters)
                        subset = torch.utils.data.Subset(dataset, random_indices)
                        random_sample_loader = torch.utils.data.DataLoader(subset, batch_size=dataset_num_clusters)
                        random_iterator = iter(random_sample_loader)
                        random_samples, _ = next(random_iterator)
                        random_samples = random_samples.to(self.device)
                        random_feature_representations = associated_feature_extractor(random_samples)

                        self.current_centroids = random_feature_representations.to(self.device)
                        current_sample_centroid_assignments = None
                    elif centroid_initialization_strategy == 'random_uniform_noise':
                        # obtain dimensions of feature representations
                        sample_loader = construct_dataset_loader(batch_size=1)
                        iterator = iter(sample_loader)
                        sample, _ = next(iterator)
                        sample = sample.to(self.device)
                        temp_feature_representations = associated_feature_extractor(sample)

                        self.current_centroids = \
                            -1.0 + 2.0 * torch.rand([dataset_num_clusters, *temp_feature_representations.shape[1:]],
                                                    device=self.device)
                        current_sample_centroid_assignments = None
                    elif centroid_initialization_strategy == 'load_from_file':
                        centroid_file_path =\
                            utils.resolve_relative_path(cluster_assignment_strategy_cfg['centroid_file_path'],
                                                        self.config_dir_path)
                        if not os.path.isfile(centroid_file_path):
                            return False, self.log(('Error: could not initialize feature-space centroids' +
                                                    'from file "%s": ' +
                                                    'file does not exist') % centroid_file_path, flush_to_file=True)
                        try:
                            load_dict = torch.load(centroid_file_path, map_location=self.device)
                            self.current_centroids = load_dict['centroids']
                        except Exception as e:
                            return False, self.log(('Error: could not initialize feature-space centroids' +
                                                    'from file "%s": %s') %
                                                   (centroid_file_path, str(e)), flush_to_file=True)

                    if self.current_centroids is not None:
                        self.log('Initialized feature-space centroids')
                    else:
                        return False, self.log('Error: could not initialize feature-space centroids',
                                               flush_to_file=True)

                phase_optimizers_grouped = {}
                if 'optimizers' in phase_cfg:
                    for optimizer_cfg in phase_cfg['optimizers']:
                        optimizer_name = optimizer_cfg.get('name', None)
                        optimizer_group_name = optimizer_cfg.get('group_name', 'default')
                        optimizer_optimizes_centroids = optimizer_cfg.get('optimizes_centroids', False)
                        optimizer_cfg_keys_to_remove = ['name', 'type', 'trained_mappings', 'trained_mapping_names',
                                                        'optimizes_centroids', 'group_name']

                        if optimizer_group_name not in phase_optimizers_grouped:
                            phase_optimizers_grouped[optimizer_group_name] = []

                        # note: create separate optimizer for each mapping/the centroids
                        for trained_mapping_name in optimizer_cfg.get('trained_mapping_names',
                                                                        optimizer_cfg.get('trained_mappings', [])):
                            mapping = next(filter(lambda c: c.name == trained_mapping_name, self.mappings), None)
                            if mapping is None:
                                return False, self.log(('Error: could not find mapping "%s" during initialization ' +
                                                        'of optimizers for phase "%s"') %
                                                       (trained_mapping_name, phase_name), flush_to_file=True)

                            optimizer_args = dict(optimizer_cfg)
                            optimizer_args['params'] = mapping.parameters()
                            optimizer_object = \
                                utils.recursive_type_init(base_namespace=torch.optim, config=optimizer_args,
                                                          additional_keys_to_remove=optimizer_cfg_keys_to_remove)

                            phase_optimizers_grouped[optimizer_group_name]\
                                .append({'name': optimizer_name,
                                         'optimizer_object': optimizer_object,
                                         'trained_mapping_name': trained_mapping_name,
                                         'optimizes_centroids': optimizer_optimizes_centroids})

                        # note: centroids must not be reinitialized
                        if optimizer_optimizes_centroids:
                            if self.current_centroids is None:
                                return False, self.log(('Error: centroids not initialized upon initialization ' +
                                                        'of centroid optimizer(s) for phase "%s"') %
                                                       phase_name, flush_to_file=True)

                            self.current_centroids.requires_grad_(True)
                            optimizer_args = dict(optimizer_cfg)
                            optimizer_args['params'] = [self.current_centroids]
                            optimizer_object = \
                                utils.recursive_type_init(base_namespace=torch.optim, config=optimizer_args,
                                                          additional_keys_to_remove=optimizer_cfg_keys_to_remove)
                            phase_optimizers_grouped[optimizer_group_name]\
                                .append({'name': optimizer_name,
                                         'optimizer_object': optimizer_object,
                                         'trained_mapping_name': None,
                                         'optimizes_centroids': optimizer_optimizes_centroids})

                # initialize optimizer lr schedulers
                phase_optimizer_lr_schedulers = []
                if 'optimizer_learning_rate_schedulers' in phase_cfg:
                    all_phase_optimizers = utils.flatten(list(phase_optimizers_grouped.values()))
                    for optimizer_lr_scheduler_cfg in phase_cfg['optimizer_learning_rate_schedulers']:
                        optimizer_name = optimizer_lr_scheduler_cfg['optimizer_name']
                        # note that there can be multiple LR schedulers with this name
                        optimizers = list(filter(lambda opt: opt['name'] == optimizer_name, all_phase_optimizers))
                        if len(optimizers) == 0:
                            return False, self.log(('Error: could not find optimizer "%s" during initialization ' +
                                                    'of optimizer learning rate schedulers for phase "%s"') %
                                                   (optimizer_name, phase_name), flush_to_file=True)
                        for optimizer in optimizers:
                            optimizer_lr_scheduler_cfg_new = dict(optimizer_lr_scheduler_cfg)
                            optimizer_lr_scheduler_cfg_new['optimizer'] = optimizer['optimizer_object']
                            lr_scheduler = utils.recursive_type_init(base_namespace=torch.optim.lr_scheduler,
                                                                     config=optimizer_lr_scheduler_cfg_new,
                                                                     additional_keys_to_remove=['optimizer_name'])
                            phase_optimizer_lr_schedulers.append({'scheduler_object': lr_scheduler})

                phase_optimizer_cycle_length = 0
                if 'optimizer_cycle' in phase_cfg:
                    phase_optimizer_cycle_length = sum([optimizer_cycle_item_cfg['num_iterations']
                                                       for optimizer_cycle_item_cfg in
                                                       phase_cfg['optimizer_cycle']])

                def get_active_optimizers(epoch_num, epoch_iter_num, phase_iter_num):
                    if phase_optimizer_cycle_length == 0:
                        return phase_optimizers_grouped
                    # phase_iter_num starts at 1
                    current_cycle_iter = (phase_iter_num - 1) % phase_optimizer_cycle_length
                    iter_cumulative_sum = 0
                    for optimizer_cycle_item_cfg in phase_cfg['optimizer_cycle']:
                        optimizer_cycle_item_num_iters = optimizer_cycle_item_cfg['num_iterations']
                        if current_cycle_iter < iter_cumulative_sum + optimizer_cycle_item_num_iters:
                            phase_optimizers_grouped_filter = {}
                            for group in phase_optimizers_grouped:
                                phase_optimizers_grouped_filter[group] = list(filter(
                                    lambda c: c['name'] in optimizer_cycle_item_cfg['active_optimizer_names'],
                                    phase_optimizers_grouped[group]))
                            return phase_optimizers_grouped_filter
                        iter_cumulative_sum += optimizer_cycle_item_num_iters
                    return {}

                # patterns only initialized *after* centroid initialization
                # separate phases should be created if a deviation from this behavior is desired
                for design_pattern_cfg in phase_design_patterns_cfg:
                    if not design_pattern_cfg.get('enabled', True):
                        continue
                    design_pattern_class = design_pattern_name_to_class(design_pattern_cfg['pattern'])
                    if design_pattern_class is None:
                        return False, self.log('Error: unknown design pattern "%s"' % design_pattern_cfg['pattern'],
                                               flush_to_file=True)

                    # initialize design pattern
                    design_pattern = design_pattern_class(self, design_pattern_cfg, phase_cfg,
                                                          dataset_cfg, dataset_num_samples, construct_dataset_loader)
                    if not design_pattern.initialization_success:
                        return False, self.log(design_pattern.initialization_err_msg, flush_to_file=True)
                    phase_design_patterns.append(design_pattern)

                phase_iter_num = 1
                real_epoch_num = 0
                real_epoch_iter_num = 0
                nested_loop_phase_exit = False
                for epoch_num in itertools.count(start=1):
                    # check if any exit criterion fulfilled
                    if nested_loop_phase_exit:  # phase exit detected in inner loop
                        break
                    # 0: epoch_iter_num
                    success, err_msg, do_exit_phase, exit_reason_msg =\
                        check_phase_exit_criteria(epoch_num, 0, real_epoch_num, real_epoch_iter_num, phase_iter_num)
                    if not success:
                        return success, self.log(err_msg, flush_to_file=True)
                    if do_exit_phase:
                        break

                    self.flush_log_to_file()

                    # only set if epoch actually used for training
                    real_epoch_num = epoch_num

                    # reinitialize batch augmentation iterators
                    for batch_augmentation_name in dataset_batch_augmentation_transform_loaders:
                        dataset_batch_augmentation_transform_iterators[batch_augmentation_name] = \
                            iter(dataset_batch_augmentation_transform_loaders[batch_augmentation_name])

                    for epoch_iter_num, iter_data in enumerate(dataset_loader, start=1):
                        if phase_has_max_total_iters and phase_iter_num > phase_max_total_iters:
                            break  # will be detected by check_phase_exit_criteria during next epoch_num iteration
                        batch_samples, batch_labels = iter_data
                        batch_samples, batch_labels = batch_samples.to(self.device), batch_labels.to(self.device)

                        batch_augmentations = {}
                        for batch_augmentation_name in dataset_batch_augmentation_transform_iterators:
                            batch_augmentation_iterator =\
                                dataset_batch_augmentation_transform_iterators[batch_augmentation_name]
                            batch_augmentation_samples, _ = next(batch_augmentation_iterator)
                            batch_augmentations[batch_augmentation_name] = batch_augmentation_samples.to(self.device)

                        if len(batch_samples) != dataset_batch_size:
                            continue

                        if phase_name in use_centroids_during_phases:
                            iterations_since_centroid_initialization += 1
                            if centroid_recalculation_interval > 0 and \
                                    iterations_since_centroid_initialization % centroid_recalculation_interval == 0:
                                iterations_since_centroid_initialization = 0
                                if centroid_recalculation_strategy == 'classical_clustering':
                                    success, err_msg =\
                                        update_centroids_by_classical_clustering(
                                            cluster_assignment_strategy_cfg
                                            ['centroid_recalculation_classical_clustering_method'],
                                            cluster_assignment_strategy_cfg.
                                                get('centroid_recalculation_classical_clustering_num_restarts', 20))
                                    if not success:
                                        return success, self.log(err_msg, flush_to_file=True)
                                # centroids may also be recalculated/modified by design patterns;
                                # in this case, no action required here

                        for design_pattern in phase_design_patterns:
                            input_samples = batch_samples
                            if len(design_pattern.batch_augmentation_names) > 0:
                                design_pattern_batch_augmentations = []
                                for augmentation_name in design_pattern.batch_augmentation_names:
                                    if augmentation_name not in dataset_batch_augmentation_transforms:
                                        return False, self.log(('Could not find batch augmentation "%s" ' +
                                                                'for design pattern "%s"') %
                                                               (augmentation_name, design_pattern.pattern_name)
                                                               , flush_to_file=True)
                                    design_pattern_batch_augmentations.append(batch_augmentations[augmentation_name])

                                if len(design_pattern_batch_augmentations) == 1:  # do not add extra dimension
                                    input_samples = design_pattern_batch_augmentations[0]
                                else:
                                    input_samples = torch.stack(design_pattern_batch_augmentations)

                            success, err_msg = design_pattern.iteration(epoch_num, epoch_iter_num, phase_iter_num,
                                                                        input_samples, batch_labels)
                            if not success:
                                return success, self.log(err_msg, flush_to_file=True)

                        active_optimizers_grouped = get_active_optimizers(epoch_num, epoch_iter_num, phase_iter_num)

                        for group_name in active_optimizers_grouped:
                            optimizer_group_active_optimizers = active_optimizers_grouped[group_name]
                            if len(optimizer_group_active_optimizers) == 0:
                                continue

                            for optimizer_cfg in optimizer_group_active_optimizers:
                                optimizer_cfg['optimizer_object'].zero_grad()

                            total_group_loss = torch.zeros(1, device=self.device)
                            for design_pattern in phase_design_patterns:
                                design_pattern_group_loss = \
                                    design_pattern.iteration_losses_grouped.get(group_name, None)
                                if design_pattern_group_loss is not None:
                                    total_group_loss += design_pattern_group_loss

                            total_group_loss.backward()

                            for optimizer_cfg in optimizer_group_active_optimizers:
                                optimizer_cfg['optimizer_object'].step()

                        if phase_performance_evaluation_interval > 0:
                            iterations_since_last_performance_evaluation += 1
                            if iterations_since_last_performance_evaluation %\
                                    phase_performance_evaluation_interval == 0:
                                # iterations_since_last_performance_evaluation reset in evaluate_performance_on_dataset
                                success, err_msg = evaluate_performance_on_dataset(epoch_num,
                                                                                   epoch_iter_num,
                                                                                   phase_iter_num)
                                if not success:
                                    return success, self.log(err_msg, flush_to_file=True)

                                self.log(
                                    utils.get_performance_string(epoch_num, epoch_iter_num,
                                                                 last_performance_evaluation_acc,
                                                                 last_performance_evaluation_nmi,
                                                                 last_performance_evaluation_ari),
                                    flush_to_file=True)

                                # check if any performance-/reassignment-based exit criteria satisfied
                                success, err_msg, do_exit_phase, exit_reason_msg =\
                                    check_phase_exit_criteria(epoch_num, epoch_iter_num,
                                                              real_epoch_num, real_epoch_iter_num,
                                                              phase_iter_num)
                                if not success:
                                    return success, self.log(err_msg, flush_to_file=True)
                                if do_exit_phase:
                                    nested_loop_phase_exit = True
                                    break

                        # only save parameters after checking whether to exit phase
                        if phase_needs_mapping_parameter_saving:
                            success, err_msg = check_save_mapping_parameters\
                                (False, epoch_num, epoch_iter_num, phase_iter_num)
                            if not success:
                                return success, self.log(err_msg, flush_to_file=True)

                        phase_iter_num += 1
                        real_epoch_iter_num = epoch_iter_num

                    for optimizer_lr_scheduler_cfg in phase_optimizer_lr_schedulers:
                        optimizer_lr_scheduler_cfg['scheduler_object'].step()

                # if at least one performance evaluation performed, report final performance on dataset
                if last_performance_evaluation_one_hot_cluster_assignments is not None:
                    self.log('Final performance on dataset "%s": ACC %.3f, NMI %.3f, ARI %.3f' %
                             (dataset_name, last_performance_evaluation_acc,
                              last_performance_evaluation_nmi, last_performance_evaluation_ari),
                             flush_to_file=True)
                else:
                    self.flush_log_to_file()

        self.log('Finished processing method "%s"' % self.name, flush_to_file=True)


def design_pattern_name_to_class(name):
    if name == 'training_feature_extractor_through_reconstruction_of_samples':
        return design_pattern.TrainingFeatureExtractorThroughReconstructionOfSamples
    elif name == 'encouraging_cluster_formation_by_minimizing_divergence_between_current_and_target_' +\
            'cluster_assignment_distribution':
        return design_pattern.\
            EncouragingClusterFormationByMinDivergenceBetweenCurrentAndTargetClusterAssignmentDistr
    elif name == 'learning_feature_representations_by_using_contrastive_learning_and_data_augmentation':
        return design_pattern.LearningFeatureRepresentationsByUsingContrastiveLearningAndDataAugmentation
    elif name == 'learning_invariance_to_transformations_by_using_assignment_statistics_vectors_' +\
            'and_data_augmentation':
        return design_pattern.\
            LearningInvarianceToTransformationsByUsingAssignmentStatisticsVectorsAndDataAugmentation
    elif name == 'preventing_cluster_degeneracy_by_maximizing_entropy_of_soft_assignments':
        return design_pattern.PreventingClusterDegeneracyByMaximizingEntropyOfSoftAssignments
    elif name == 'training_feature_extractor_by_using_adversarial_interpolation':
        return design_pattern.TrainingFeatureExtractorByUsingAdversarialInterpolation
    elif name == 'facilitating_training_of_feature_extractor_by_using_layer_wise_pretraining':
        return design_pattern.FacilitatingTrainingOfFeatureExtractorByUsingLayerWisePretraining
    elif name == 'encouraging_cluster_formation_by_reinforcing_current_assignment_of_samples_to_clusters'\
            or name == 'encouraging_cluster_formation_by_reinforcing_current_assignments_of_samples_to_clusters':
        return design_pattern.EncouragingClusterFormationByReinforcingCurrentAssignmentOfSamplesToClusters
    elif name == 'encouraging_cluster_formation_by_minimizing_divergence_between_current_and_target_' +\
                 'cluster_assignment_distribution':
        return design_pattern.EncouragingClusterFormationByMinDivergenceBetweenCurrentAndTargetClusterAssignmentDistr
    return None

