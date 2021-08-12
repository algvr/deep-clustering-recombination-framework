import torch.nn as nn
import torch.nn.functional
import utils


class Mapping(nn.Module):
    def __init__(self, mapping_cfg, method):
        super().__init__()
        self.config = mapping_cfg
        self.type = self.config['type']
        self.name = self.config.get('name', self.type)
        self.method = method

        layer_initialization_cfg_list = mapping_cfg.get('layer_initialization', None)
        top_level_layer_list = []
        self.layers = {}
        layers_cfg = self.config['layers']

        layer_index = 0

        # needed for adding/initializing nested layers:
        def add_and_initialize_layer(constructor_function, positional_arguments, args):
            nonlocal layer_index
            layer_name = args.pop('name', layer_index)
            layer = constructor_function(*positional_arguments, **args)
            self.layers[layer_name] = layer

            if layer_initialization_cfg_list is not None:
                for layer_initialization_cfg in filter(lambda cfg: layer_name in cfg['layer_names'],
                                                       layer_initialization_cfg_list):
                    if not layer_initialization_cfg['type'].endswith('_'):
                        layer_initialization_cfg['type'] += '_'
                    layer_initialization_cfg['tensor'] = layer.__getattr__(layer_initialization_cfg['weight_type']).data
                    utils.recursive_type_init(base_namespace=nn.init, config=layer_initialization_cfg,
                                              additional_keys_to_remove=['layer_names', 'weight_type'])

            layer_index += 1
            return layer

        for layer_cfg in layers_cfg:
            top_level_layer_list.append(utils.recursive_type_init(base_namespace=nn, config=layer_cfg,
                                                                  additional_keys_to_remove=[],
                                                                  callback=add_and_initialize_layer))

        self.net = nn.Sequential(*top_level_layer_list)

        self.net_without_softmax_layer = None
        if 'softmax_layer_name' in self.config:
            softmax_layer_name = self.config['softmax_layer_name']
            softmax_layer = self.layers.get(softmax_layer_name, None)
            self.net_without_softmax_layer = nn.Sequential(*[layer_object
                                                             for layer_object in top_level_layer_list
                                                             if layer_object != softmax_layer])

        self.pre_processing_functions = []
        self.post_processing_functions = []
        pre_processing_function_cfgs = self.config.get('pre_processing_functions', [])
        post_processing_function_cfgs = self.config.get('post_processing_functions', [])
        pre_post_processing_function_cfgs = [*map(lambda cfg: {'type': 'pre', 'config': cfg},
                                                  pre_processing_function_cfgs),
                                             *map(lambda cfg: {'type': 'post', 'config': cfg},
                                                  post_processing_function_cfgs)]
        for pre_post_processing_function_cfg_wrapped in pre_post_processing_function_cfgs:
            function_type = pre_post_processing_function_cfg_wrapped['type']
            function_cfg = pre_post_processing_function_cfg_wrapped['config']
            function_name = function_cfg['function_name']
            input_argument_name = function_cfg['input_argument_name']
            function_args = utils.recursive_type_init(torch.nn, function_cfg, ['function_name', 'input_argument_name'])
            function_object = utils.getattr_recursive(torch.nn.functional, function_name)
            function_object_wrapper = \
                lambda x, f=function_object, a=function_args, n=input_argument_name: f(**{**a, n: x})
            if function_type == 'pre':
                self.pre_processing_functions.append(function_object_wrapper)
            else:
                self.post_processing_functions.append(function_object_wrapper)

    def get_prior_mapping_chain(self):
        chain = []
        next_mapping = self.get_prior_mapping()
        while next_mapping is not None:
            chain.append(next_mapping)
            next_mapping = next_mapping.get_prior_mapping()
        return chain

    def get_associated_feature_extractor(self):
        if self.type != 'feature_space_classifier':
            return None

        # choose first declared feature extractor if none specified
        feature_extractor_name = self.config.get('associated_feature_extractor_name', None)
        if feature_extractor_name is None:
            return next(filter(lambda c: c.type == 'feature_extractor', self.method.mappings), None)
        else:
            return next(filter(lambda c: c.name == feature_extractor_name, self.method.mappings), None)

    def get_prior_mapping(self):
        # chaining mappings
        if 'prior_mapping_name' not in self.config:
            return None

        # choose first declared feature extractor if none specified
        prior_mapping_name = self.config['prior_mapping_name']
        return next(filter(lambda c: c.name == prior_mapping_name, self.method.mappings), None)

    def pre_process(self, x):
        if 'prior_mapping_name' in self.config:
            # chaining mappings
            prior_mapping = next(filter(lambda c: c.name == self.config['prior_mapping_name'],
                                          self.method.mappings), None)
            if prior_mapping is None:
                raise Exception('Associated prior mapping "%s" of mapping "%s" not found' %
                                (self.config['prior_mapping_name'], self.name))
            x = prior_mapping(x)
        for pre_processing_function in self.pre_processing_functions:
            x = pre_processing_function(x)
        return x

    def post_process(self, x):
        for post_processing_function in self.post_processing_functions:
            x = post_processing_function(x)
        return x

    def get_logits(self, x):
        if self.net_without_softmax_layer is not None:
            return self.post_process(self.net_without_softmax_layer(self.pre_process(x)))
        else:
            return self.forward(x)

    def forward(self, x):
        return self.post_process(self.net(self.pre_process(x)))
