# deep-clustering-recombination-framework

Recombination framework for deep-learning–based clustering methods, based on "An ontology for systematization and recombination of deep-learning–based clustering methods".

## How to use

1. Configure one or more deep-learning–⁠based clustering methods using JSON files following the JSON schema found in ``schema/method.json``. Instructions can be found in the ``description`` properties of the respective JSON properties and objects in the schema files. For definitions of terms found therein and further elaborations, see the aforementioned ontology.
2. Run ``main.py`` with the configuration files of the methods to process as arguments (all paths are considered to be relative to the location of ``main.py``). Any directories passed as arguments will be searched non-recursively, and any found JSON files will be processed.

## Arguments for ``main.py``
``--one_log_file``: path to single log file to use (uses separate log files for each method to process if omitted)

``--no_log_files``: do not create log files

``--no_log_timestamps``: do not prefix log messages with timestamps

``--resume_on_error``: if an exception is raised while processing a method, process the next method instead of crashing

## Cluster assignment strategies

The following cluster assignment strategies are currently implemented:
* based on output of sample-space classifier
* based on output of feature-space classifier
* classical clustering in feature space after training (using k-means)
* based on feature-space centroids calculated during training (using soft assignments)

## Trainable mappings

In general, trainable mappings with layers from any class in the ``torch.nn`` module of the ``torch`` Python package can be constructed.

## Design patterns
JSON schema definitions of design patterns, as described in the ontology, can be found in ``schema/design_patterns``. Currently, the following design patterns are implemented:
* training a feature extractor through reconstruction of samples
* training a feature extractor by using adversarial interpolation
* facilitating the training of a feature extractor by using layer-wise pretraining (variant using denoising autoencoder)
* learning transformation-invariant feature representations by using contrastive learning and data augmentation (variant using SimCLR)
* learning invariance of soft assignments to transformations by using assignment statistics vectors and data augmentation
* encouraging cluster formation by minimizing the divergence between the current cluster assignment distribution and a derived target distribution
* encouraging cluster formation by reinforcing the current assignment of samples to clusters (variants in feature space, in sample space using a decoder, and based on soft assignments)
* preventing cluster degeneracy by maximizing the entropy of soft assignments

## Datasets

Note that the recombination framework is currently limited to processing datasets in the ``torchvision.datasets`` module of the ``torchvision`` Python package.

## Exemplary method configurations
Some configurations of deep-learning–based clustering methods, as discussed in "An ontology for systematization and recombination of deep-learning–based clustering methods", can be found in subdirectories of the ``configurations`` directory. Details can be found in the respective method's configuration file.
* ``dec_cc_hybrid``: This method uses the standard ``784-500-500-2000-10`` encoder (introduced in https://arxiv.org/abs/1511.06335) as its feature extractor. It computes soft assignments based on a Student's-t–kernel measuring the similarity between feature representations and feature-space centroids as its cluster assignment strategy. Furthermore, it uses the "training a feature extractor through reconstruction of samples" design pattern during its pretraining phase, and the "learning transformation-invariant feature representations by using contrastive learning and data augmentation", "learning invariance of soft assignments to transformations by using assignment statistics vectors and data augmentation", "preventing cluster degeneracy by maximizing the entropy of soft assignments" (all three from the method in https://arxiv.org/abs/2009.09687), and "encouraging cluster formation by minimizing the divergence between the current cluster assignment distribution and a derived target distribution" design patterns during its finetuning phase.\
Achieves a performance of ACC 0.978 (97.8 ± 0.2), NMI 0.944 (94.4 ± 1.0), ARI 0.952 (95.2 ± 0.4) evaluated on MNIST-Test after training on MNIST-Train. Further achieves a performance of ACC 0.583 (58.3 ± 1.5), NMI 0.633 (63.3 ± 0.6), ARI 0.470 (47.0 ± 1.5) on Fashion-MNIST-Test after training on Fashion-MNIST-Train. The values in parentheses indicate means and standard deviations over 10 runs. Configuration files are provided both for evaluation on MNIST-Test and on Fashion-MNIST-Test.
* ``deep_k_means``: Recreation of the DKM-a method (introduced in https://arxiv.org/abs/1806.10069) on MNIST. This method uses the aforementioned ``784-500-500-2000-10`` encoder as its feature extractor. It computes soft assignments based on a Gaussian kernel measuring the similarity between feature representations and feature-space centroids as its cluster assignment strategy, using an annealed inverse temperature as described in https://arxiv.org/abs/1806.10069. Furthermore, it uses the feature-space–based variant of the "encouraging cluster formation by reinforcing the current assignment of samples to clusters" design pattern, as well as the "training a feature extractor through reconstruction of samples" design pattern during its single training phase.
* ``layer_wise_pretraining``: A method trained using solely the denoising-autoencoder–based variant of the "facilitating the training of a feature extractor by using layer-wise pretraining" design pattern. The method uses the "classical clustering in feature space after training" cluster assignment strategy, using the aforementioned ``784-500-500-2000-10`` encoder as its feature extractor. Intended to test the correct implementation of the "facilitating the training of a feature extractor by using layer-wise pretraining" design pattern, and can be used as a basis for the construction of a method following a pretraining-finetuning training schedule.
* ``adversarial_interpolation_pretraining``: A method trained using solely the "training a feature extractor by using adversarial interpolation" design pattern, using the aforementioned ``784-500-500-2000-10`` encoder as its feature extractor. The method uses the "classical clustering in feature space after training" cluster assignment strategy. Intended to test the correct implementation of the "training a feature extractor by using adversarial interpolation" design pattern, and can be used as a basis for the construction of a method following a pretraining-finetuning training schedule.