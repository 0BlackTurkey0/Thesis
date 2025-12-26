# Keras3 API
import keras

class Conv(keras.layers.Layer):
    def __init__(self, filters: int, kernel_size=3, strides=2, momentum=0.9, epsilon=1e-5, activation=keras.activations.gelu, **kwargs):
        super(Conv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.momentum = momentum
        self.epsilon = epsilon
        self.activation = activation
        
    def build(self, input_shape):
        self.conv_layers = []

        conv = keras.layers.Conv2D(self.filters, self.kernel_size, strides=self.strides, padding="same", use_bias=False)
        norm = keras.layers.BatchNormalization(momentum=self.momentum, epsilon=self.epsilon)
        act = keras.layers.Activation(self.activation)
        self.conv_layers += [conv, norm, act]
        
    def call(self, inputs):
        # Conv Branch
        conv = inputs
        for conv_layer in self.conv_layers:
            conv = conv_layer(conv)
        result = conv
        return result
        
    def get_config(self):
        config = super(Conv, self).get_config()
        config["filters"] = self.filters
        config["kernel_size"] = self.kernel_size
        config["strides"] = self.strides
        config["momentum"] = self.momentum
        config["epsilon"] = self.epsilon
        config["activation"] = self.activation
        return config

class MBConv(keras.layers.Layer):
    '''
    Depthwise Separable Convolution (Depthwise + Pointwise Convolution): https://arxiv.org/pdf/1704.04861,
    Squeeze-and-Excitation Networks (SENet): https://arxiv.org/pdf/1709.01507,
    Mobile Inverted Bottleneck Convolution (MBConv): https://arxiv.org/pdf/1801.04381,
    MBConv with SENet (MnasNet): https://arxiv.org/pdf/1807.11626
    '''
    
    def __init__(self, filters: int, kernel_size=3, strides=2, expansion_factor=4, se_reduction_ratio=4, momentum=0.9, epsilon=1e-5, activation=keras.activations.gelu, drop_rate=0., sd_drop_rate=0., **kwargs):
        super(MBConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.expansion_factor = expansion_factor
        self.se_reduction_ratio = se_reduction_ratio
        self.momentum = momentum
        self.epsilon = epsilon
        self.activation = activation
        self.drop_rate = drop_rate
        self.sd_drop_rate = sd_drop_rate
        
    def build(self, input_shape):
        self.identity_layers = []
        self.residual_layers = []

        # Pre-Norm
        pre_norm = keras.layers.BatchNormalization(momentum=self.momentum, epsilon=self.epsilon)
        self.residual_layers += [pre_norm]
        
        # Down-Sampling
        channels = input_shape[-1]
        if self.strides > 1:
            pool = keras.layers.MaxPool2D(pool_size=self.strides + (self.strides + 1) % 2, strides=self.strides, padding="same")
            self.identity_layers += [pool]
        if channels != self.filters:
            proj = keras.layers.Conv2D(self.filters, 1, use_bias=False)
            self.identity_layers += [proj]
        
        # Expansion Pointwise Convolution
        if self.expansion_factor > 1:
            channels *= self.expansion_factor
            conv = keras.layers.Conv2D(channels, 1, use_bias=False)
            norm = keras.layers.BatchNormalization(momentum=self.momentum, epsilon=self.epsilon)
            act = keras.layers.Activation(self.activation)
            self.residual_layers += [conv, norm, act]
        
        # Depthwise Convolution
        dw_conv = keras.layers.DepthwiseConv2D(self.kernel_size, strides=self.strides, padding="same", use_bias=False)
        norm = keras.layers.BatchNormalization(momentum=self.momentum, epsilon=self.epsilon)
        act = keras.layers.Activation(self.activation)
        self.residual_layers += [dw_conv, norm, act]
        
        # Squeeze-and-Excitation Networks
        if self.se_reduction_ratio > 1:
            reduced_channels = max(1, channels // self.se_reduction_ratio)
            squeeze = keras.layers.GlobalAveragePooling2D(keepdims=True)
            excitation_reduce = keras.layers.Dense(reduced_channels, activation=keras.activations.relu)
            excitation_expand = keras.layers.Dense(channels, activation=keras.activations.sigmoid)
            se = keras.layers.Lambda(lambda x: x * excitation_expand(excitation_reduce(squeeze(x))))
            self.residual_layers += [se]
        
        # Projection Pointwise Convolution
        conv = keras.layers.Conv2D(self.filters, 1, use_bias=False)
        norm = keras.layers.BatchNormalization(momentum=self.momentum, epsilon=self.epsilon)
        drop = keras.layers.Dropout(rate=self.drop_rate)
        self.residual_layers += [conv, norm, drop]

        # Stochastic Depth
        self.stoc_depth = keras.layers.Dropout(rate=self.sd_drop_rate, noise_shape=(None, 1, 1, 1))
        
    def call(self, inputs):
        # Identity Branch
        identity = inputs
        for identity_layer in self.identity_layers:
            identity = identity_layer(identity)

        # Residual Branch
        residual = inputs
        for residual_layer in self.residual_layers:
            residual = residual_layer(residual)

        # Skip Connection
        result = identity + self.stoc_depth(residual)
        return result
        
    def get_config(self):
        config = super(MBConv, self).get_config()
        config["filters"] = self.filters
        config["kernel_size"] = self.kernel_size
        config["strides"] = self.strides
        config["expansion_factor"] = self.expansion_factor
        config["se_reduction_ratio"] = self.se_reduction_ratio
        config["momentum"] = self.momentum
        config["epsilon"] = self.epsilon
        config["activation"] = self.activation
        config["drop_rate"] = self.drop_rate
        config["sd_drop_rate"] = self.sd_drop_rate
        return config
    
class TFMRel(keras.layers.Layer):
    '''
    Multi-Head Self-Attention with Relative Position Bias (Swin Transformer): https://arxiv.org/pdf/2103.14030
    '''

    def __init__(self, filters: int, head_dim=32, strides=2, expansion_factor=4, epsilon=1e-5, activation=keras.activations.gelu, drop_rate=0., sd_drop_rate=0., **kwargs):
        super(TFMRel, self).__init__(**kwargs)
        self.filters = filters
        self.head_dim = head_dim
        self.strides = strides
        self.expansion_factor = expansion_factor
        self.epsilon = epsilon
        self.activation = activation
        self.drop_rate = drop_rate
        self.sd_drop_rate = sd_drop_rate
        
    def build(self, input_shape):
        self.att_identity_layers = []
        self.att_residual_layers = []
        self.ffn_residual_layers = []

        # Pre-Norm
        pre_norm = keras.layers.LayerNormalization(epsilon=self.epsilon)
        self.att_residual_layers += [pre_norm]
        pre_norm = keras.layers.LayerNormalization(epsilon=self.epsilon)
        self.ffn_residual_layers += [pre_norm]

        # Down-Sampling
        channels = input_shape[-1]
        if self.strides > 1:
            pool = keras.layers.MaxPool2D(pool_size=self.strides + (self.strides + 1) % 2, strides=self.strides, padding="same")
            self.att_identity_layers += [pool]
            pool = keras.layers.MaxPool2D(pool_size=self.strides + (self.strides + 1) % 2, strides=self.strides, padding="same")
            self.att_residual_layers += [pool]
        if channels != self.filters:
            proj = keras.layers.Conv2D(self.filters, 1, use_bias=False)
            self.att_identity_layers += [proj]
            
        # Relative Position Bias
        heads = -(self.filters // -self.head_dim) # ceiling
        height = -(input_shape[1] // -self.strides) # ceiling
        width = -(input_shape[2] // -self.strides) # ceiling
        coords_h = keras.ops.arange(height)
        coords_w = keras.ops.arange(width)
        coords = keras.ops.stack(keras.ops.meshgrid(coords_h, coords_w, indexing = "ij"))
        coords = keras.ops.reshape(coords, (2, -1))
        relative_coords = keras.ops.expand_dims(coords, axis=-1) - keras.ops.expand_dims(coords, axis=-2)
        relative_h = (relative_coords[0] + (height - 1)) * (2 * width - 1)
        relative_w = (relative_coords[1] + (width - 1))
        relative_coords = keras.ops.stack((relative_h, relative_w))
        relative_position = keras.ops.sum(relative_coords, axis=0)
        self.relative_position_index = keras.ops.reshape(relative_position, (-1,))
        self.relative_position_table = self.add_weight(shape=((2 * height - 1) * (2 * width - 1), heads), trainable=True)
        self.relative_position_bias = lambda: keras.ops.expand_dims(keras.ops.transpose(keras.ops.reshape(keras.ops.take(self.relative_position_table, self.relative_position_index, axis=0), (height * width, height * width, -1)), [2, 0, 1]), axis=0)

        # Relative Positional Multi-Head Self-Attention
        pre_reshape = keras.layers.Reshape((-1, channels))
        qkv = keras.layers.Dense(3 * heads * self.head_dim, use_bias=False, kernel_initializer=keras.initializers.GlorotNormal())
        head_reshape = keras.layers.Reshape((-1, 3, heads, self.head_dim))
        head_permute = keras.layers.Permute((2, 3, 1, 4))
        split = keras.layers.Lambda(lambda x: keras.ops.stack(keras.ops.unstack(x, axis=1)))
        self.att_residual_layers += [pre_reshape, qkv, head_reshape, head_permute, split]

        softmax = keras.layers.Softmax()
        drop = keras.layers.Dropout(rate=self.drop_rate)
        attention = keras.layers.Lambda(lambda x: keras.ops.matmul(drop(softmax(keras.ops.matmul(x[0], keras.ops.transpose(x[1], [0, 1, 3, 2])) / keras.ops.cast(keras.ops.shape(x[1])[-1], "float32") + self.relative_position_bias())), x[2]))
        self.att_residual_layers += [attention]

        out_permute = keras.layers.Permute((2, 1, 3))
        out_reshape = keras.layers.Reshape((-1, heads * self.head_dim))
        concat = keras.layers.Dense(self.filters, kernel_initializer=keras.initializers.GlorotNormal())
        drop = keras.layers.Dropout(rate=self.drop_rate)
        post_reshape = keras.layers.Reshape((height, width, self.filters))
        self.att_residual_layers += [out_permute, out_reshape, concat, drop, post_reshape]

        # Feed Forward Network
        dense1 = keras.layers.Dense(self.filters * self.expansion_factor, activation=self.activation)
        dense2 = keras.layers.Dense(self.filters)
        self.ffn_residual_layers += [dense1, dense2]
        
        # Stochastic Depth
        self.att_stoc_depth = keras.layers.Dropout(rate=self.sd_drop_rate, noise_shape=(None, 1, 1, 1))
        self.ffn_stoc_depth = keras.layers.Dropout(rate=self.sd_drop_rate, noise_shape=(None, 1, 1, 1))
    
    def call(self, inputs):
        # Attention Identity Branch
        att_identity = inputs
        for att_identity_layer in self.att_identity_layers:
            att_identity = att_identity_layer(att_identity)

        # Attention Residual Branch
        att_residual = inputs
        for att_residual_layer in self.att_residual_layers:
            att_residual = att_residual_layer(att_residual)

        # Attention Skip Connection
        ffn_identity = att_identity + self.att_stoc_depth(att_residual)

        # FFN Residual Branch
        ffn_residual = ffn_identity
        for ffn_residual_layer in self.ffn_residual_layers:
            ffn_residual = ffn_residual_layer(ffn_residual)

        # FFN Skip Connection
        result = ffn_identity + self.ffn_stoc_depth(ffn_residual)
        return result
        
    def get_config(self):
        config = super(TFMRel, self).get_config()
        config["filters"] = self.filters
        config["head_dim"] = self.head_dim
        config["strides"] = self.strides
        config["expansion_factor"] = self.expansion_factor
        config["epsilon"] = self.epsilon
        config["activation"] = self.activation
        config["drop_rate"] = self.drop_rate
        config["sd_drop_rate"] = self.sd_drop_rate
        return config

class CoAtNet:
    '''
    Convolution and Attention for All Data Sizes (CoAtNet): https://arxiv.org/pdf/2106.04803
    '''
    
    class Stage:
        def __init__(self, block: str, layers: int, filters: int, strides: int, drop_rate: float=0., sd_drop_rate: float=0.):
            self.block = block
            self.layers = layers
            self.filters = filters
            self.strides = strides
            self.drop_rate = drop_rate
            self.sd_drop_rate = sd_drop_rate

    @staticmethod
    def get_model(input_shape, stages: list[Stage], classes: int=None, kernel_size=3, head_dim=32, expansion_factor=4, se_reduction_ratio=4, momentum=0.9, epsilon=1e-5, activation=keras.activations.gelu, drop_rate=0.):
        inputs = keras.layers.Input(input_shape)

        coatnet = inputs

        # Stages
        for stage in stages:
            func, kwargs = None, None
            block = stage.block
            layers = stage.layers
            filters = stage.filters
            strides = stage.strides # reduce the spatial size by strides at the beginning of each stage
            sd_drop_rate = stage.sd_drop_rate
            if block == 'S':
                func = Conv
                kwargs = {
                    "filters": filters, "kernel_size": kernel_size, "strides": strides, "momentum": momentum, 
                    "epsilon": epsilon, "activation": activation
                }
            elif block == 'C':
                func = MBConv
                kwargs = {
                    "filters": filters, "kernel_size": kernel_size, "strides": strides, "expansion_factor": expansion_factor, 
                    "se_reduction_ratio": se_reduction_ratio, "momentum": momentum, "epsilon": epsilon, 
                    "activation": activation, "drop_rate": drop_rate, "sd_drop_rate": sd_drop_rate
                }
            elif block == 'T':
                func = TFMRel
                kwargs = {
                    "filters": filters, "head_dim": head_dim, "strides": strides, "expansion_factor": expansion_factor, 
                    "epsilon": epsilon, "activation": activation, "drop_rate": drop_rate, "sd_drop_rate": sd_drop_rate
                }
            else:
                raise ValueError(f"Unknown Stage Block({block})")
            coatnet = func(**kwargs)(coatnet)
            kwargs["strides"] = 1 # keep the spatial size at the follow-up of each stage
            for _ in range(layers - 1):
                coatnet = func(**kwargs)(coatnet)

        # Classifier
        if classes is not None:
            coatnet = keras.layers.GlobalAveragePooling2D()(coatnet)
            coatnet = keras.layers.Dense(classes, activation=keras.activations.softmax)(coatnet)
            
        outputs = coatnet

        model = keras.Model(inputs, outputs)
        return model

    @classmethod
    def coatnet0(cls, input_shape, classes: int=None, kernel_size=3, head_dim=32, expansion_factor=4, se_reduction_ratio=4, momentum=0.9, epsilon=1e-5, activation=keras.activations.gelu, drop_rate=0.):
        stages = [
            cls.Stage('S', layers=2, filters=64, strides=2),
            cls.Stage('C', layers=2, filters=96, strides=2, sd_drop_rate=0.1),
            cls.Stage('C', layers=3, filters=192, strides=2, sd_drop_rate=0.2),
            cls.Stage('T', layers=5, filters=384, strides=2, sd_drop_rate=0.3),
            cls.Stage('T', layers=2, filters=768, strides=2, sd_drop_rate=0.2),
        ]
        model = cls.get_model(input_shape, stages, classes, kernel_size, head_dim, expansion_factor, se_reduction_ratio, momentum, epsilon, activation, drop_rate)
        return model

    @classmethod
    def coatnet1(cls, input_shape, classes: int=None, kernel_size=3, head_dim=32, expansion_factor=4, se_reduction_ratio=4, momentum=0.9, epsilon=1e-5, activation=keras.activations.gelu, drop_rate=0.):
        stages = [
            cls.Stage('S', layers=2, filters=64, strides=2),
            cls.Stage('C', layers=2, filters=96, strides=2, sd_drop_rate=0.1),
            cls.Stage('C', layers=6, filters=192, strides=2, sd_drop_rate=0.3),
            cls.Stage('T', layers=14, filters=384, strides=2, sd_drop_rate=0.4),
            cls.Stage('T', layers=2, filters=768, strides=2, sd_drop_rate=0.2),
        ]
        model = cls.get_model(input_shape, stages, classes, kernel_size, head_dim, expansion_factor, se_reduction_ratio, momentum, epsilon, activation, drop_rate)
        return model

    @classmethod
    def coatnet2(cls, input_shape, classes: int=None, kernel_size=3, head_dim=32, expansion_factor=4, se_reduction_ratio=4, momentum=0.9, epsilon=1e-5, activation=keras.activations.gelu, drop_rate=0.):
        stages = [
            cls.Stage('S', layers=2, filters=128, strides=2),
            cls.Stage('C', layers=2, filters=128, strides=2, sd_drop_rate=0.1),
            cls.Stage('C', layers=6, filters=256, strides=2, sd_drop_rate=0.3),
            cls.Stage('T', layers=14, filters=512, strides=2, sd_drop_rate=0.5),
            cls.Stage('T', layers=2, filters=1024, strides=2, sd_drop_rate=0.3),
        ]
        model = cls.get_model(input_shape, stages, classes, kernel_size, head_dim, expansion_factor, se_reduction_ratio, momentum, epsilon, activation, drop_rate)
        return model

    @classmethod
    def coatnet3(cls, input_shape, classes: int=None, kernel_size=3, head_dim=32, expansion_factor=4, se_reduction_ratio=4, momentum=0.9, epsilon=1e-5, activation=keras.activations.gelu, drop_rate=0.):
        stages = [
            cls.Stage('S', layers=2, filters=192, strides=2),
            cls.Stage('C', layers=2, filters=192, strides=2, sd_drop_rate=0.1),
            cls.Stage('C', layers=6, filters=384, strides=2, sd_drop_rate=0.3),
            cls.Stage('T', layers=14, filters=768, strides=2, sd_drop_rate=0.6),
            cls.Stage('T', layers=2, filters=1536, strides=2, sd_drop_rate=0.4),
        ]
        model = cls.get_model(input_shape, stages, classes, kernel_size, head_dim, expansion_factor, se_reduction_ratio, momentum, epsilon, activation, drop_rate)
        return model

    @classmethod
    def coatnet4(cls, input_shape, classes: int=None, kernel_size=3, head_dim=32, expansion_factor=4, se_reduction_ratio=4, momentum=0.9, epsilon=1e-5, activation=keras.activations.gelu, drop_rate=0.):
        stages = [
            cls.Stage('S', layers=2, filters=192, strides=2),
            cls.Stage('C', layers=2, filters=192, strides=2, sd_drop_rate=0.1),
            cls.Stage('C', layers=12, filters=384, strides=2, sd_drop_rate=0.5),
            cls.Stage('T', layers=28, filters=768, strides=2, sd_drop_rate=0.7),
            cls.Stage('T', layers=2, filters=1536, strides=2, sd_drop_rate=0.4),
        ]
        model = cls.get_model(input_shape, stages, classes, kernel_size, head_dim, expansion_factor, se_reduction_ratio, momentum, epsilon, activation, drop_rate)
        return model

    @classmethod
    def coatnet5(cls, input_shape, classes: int=None, kernel_size=3, head_dim=64, expansion_factor=4, se_reduction_ratio=4, momentum=0.9, epsilon=1e-5, activation=keras.activations.gelu, drop_rate=0.):
        stages = [
            cls.Stage('S', layers=2, filters=192, strides=2),
            cls.Stage('C', layers=2, filters=256, strides=2, sd_drop_rate=0.1),
            cls.Stage('C', layers=12, filters=512, strides=2, sd_drop_rate=0.6),
            cls.Stage('T', layers=28, filters=1280, strides=2, sd_drop_rate=0.8),
            cls.Stage('T', layers=2, filters=2048, strides=2, sd_drop_rate=0.5),
        ]
        model = cls.get_model(input_shape, stages, classes, kernel_size, head_dim, expansion_factor, se_reduction_ratio, momentum, epsilon, activation, drop_rate)
        return model

    @classmethod
    def coatnet6(cls, input_shape, classes: int=None, kernel_size=3, head_dim=128, expansion_factor=4, se_reduction_ratio=4, momentum=0.9, epsilon=1e-5, activation=keras.activations.gelu, drop_rate=0.):
        stages = [
            cls.Stage('S', layers=2, filters=192, strides=2),
            cls.Stage('C', layers=2, filters=192, strides=2, sd_drop_rate=0.1),
            cls.Stage('C', layers=4, filters=384, strides=2, sd_drop_rate=0.3),
            cls.Stage('C', layers=8, filters=768, strides=2, sd_drop_rate=0.5),
            cls.Stage('T', layers=42, filters=1536, strides=1, sd_drop_rate=0.9),
            cls.Stage('T', layers=2, filters=2048, strides=2, sd_drop_rate=0.5),
        ]
        model = cls.get_model(input_shape, stages, classes, kernel_size, head_dim, expansion_factor, se_reduction_ratio, momentum, epsilon, activation, drop_rate)
        return model

    @classmethod
    def coatnet7(cls, input_shape, classes: int=None, kernel_size=3, head_dim=128, expansion_factor=4, se_reduction_ratio=4, momentum=0.9, epsilon=1e-5, activation=keras.activations.gelu, drop_rate=0.):
        stages = [
            cls.Stage('S', layers=2, filters=192, strides=2),
            cls.Stage('C', layers=2, filters=256, strides=2, sd_drop_rate=0.1),
            cls.Stage('C', layers=4, filters=512, strides=2, sd_drop_rate=0.4),
            cls.Stage('C', layers=8, filters=1024, strides=2, sd_drop_rate=0.6),
            cls.Stage('T', layers=42, filters=2048, strides=1, sd_drop_rate=0.9),
            cls.Stage('T', layers=2, filters=3072, strides=2, sd_drop_rate=0.5),
        ]
        model = cls.get_model(input_shape, stages, classes, kernel_size, head_dim, expansion_factor, se_reduction_ratio, momentum, epsilon, activation, drop_rate)
        return model