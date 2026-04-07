use burn::{
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{
        Tensor,
        activation::{relu, sigmoid, softmax, tanh},
        backend::Backend,
    },
};

#[derive(Clone, Debug)]
pub struct EcapaTdnnConfig {
    pub channels: usize,
    pub feat_dim: usize,
    pub embed_dim: usize,
    pub global_context_att: bool,
}

impl Default for EcapaTdnnConfig {
    fn default() -> Self {
        Self {
            channels: 512,
            feat_dim: 80,
            embed_dim: 192,
            global_context_att: true,
        }
    }
}

impl EcapaTdnnConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EcapaTdnn<B> {
        EcapaTdnn::new(self, device)
    }
}

#[derive(Module, Debug)]
pub struct Conv1dReluBn<B: Backend> {
    pub conv: Conv1d<B>,
    pub bn: BatchNorm<B>,
}

impl<B: Backend> Conv1dReluBn<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        bias: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            conv: Conv1dConfig::new(in_channels, out_channels, kernel_size)
                .with_stride(stride)
                .with_padding(PaddingConfig1d::Explicit(padding))
                .with_dilation(dilation)
                .with_bias(bias)
                .init(device),
            bn: BatchNormConfig::new(out_channels).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.bn.forward(relu(self.conv.forward(x)))
    }
}

#[derive(Module, Debug)]
pub struct Res2Conv1dReluBn<B: Backend> {
    pub convs: Vec<Conv1d<B>>,
    pub bns: Vec<BatchNorm<B>>,
    pub scale: usize,
    pub width: usize,
    pub nums: usize,
}

impl<B: Backend> Res2Conv1dReluBn<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        bias: bool,
        scale: usize,
        device: &B::Device,
    ) -> Self {
        assert_eq!(
            channels % scale,
            0,
            "{channels} must be divisible by {scale}"
        );

        let width = channels / scale;
        let nums = if scale == 1 { scale } else { scale - 1 };
        let mut convs = Vec::with_capacity(nums);
        let mut bns = Vec::with_capacity(nums);

        for _ in 0..nums {
            convs.push(
                Conv1dConfig::new(width, width, kernel_size)
                    .with_stride(stride)
                    .with_padding(PaddingConfig1d::Explicit(padding))
                    .with_dilation(dilation)
                    .with_bias(bias)
                    .init(device),
            );
            bns.push(BatchNormConfig::new(width).init(device));
        }

        Self {
            convs,
            bns,
            scale,
            width,
            nums,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut splits = Vec::with_capacity(self.scale);
        for idx in 0..self.scale {
            splits.push(x.clone().narrow(1, idx * self.width, self.width));
        }

        let mut out = Vec::with_capacity(self.scale);
        let mut state = splits[0].clone();
        for (idx, (conv, bn)) in self.convs.iter().zip(self.bns.iter()).enumerate() {
            if idx >= 1 {
                state = state + splits[idx].clone();
            }
            state = bn.forward(relu(conv.forward(state)));
            out.push(state.clone());
        }
        if self.scale != 1 {
            out.push(splits[self.nums].clone());
        }

        Tensor::cat(out, 1)
    }
}

#[derive(Module, Debug)]
pub struct SeConnect<B: Backend> {
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
}

impl<B: Backend> SeConnect<B> {
    pub fn new(channels: usize, se_bottleneck_dim: usize, device: &B::Device) -> Self {
        Self {
            linear1: LinearConfig::new(channels, se_bottleneck_dim).init(device),
            linear2: LinearConfig::new(se_bottleneck_dim, channels).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let pooled = x.clone().mean_dim(2).squeeze_dims::<2>(&[2_isize]);
        let pooled = relu(self.linear1.forward(pooled));
        let pooled = sigmoid(self.linear2.forward(pooled)).unsqueeze_dims::<3>(&[2usize]);

        x * pooled
    }
}

#[derive(Module, Debug)]
pub struct SeRes2Block<B: Backend> {
    pub pre: Conv1dReluBn<B>,
    pub res2: Res2Conv1dReluBn<B>,
    pub post: Conv1dReluBn<B>,
    pub se: SeConnect<B>,
}

impl<B: Backend> SeRes2Block<B> {
    pub fn new(
        channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        scale: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            pre: Conv1dReluBn::new(channels, channels, 1, 1, 0, 1, true, device),
            res2: Res2Conv1dReluBn::new(
                channels,
                kernel_size,
                stride,
                padding,
                dilation,
                true,
                scale,
                device,
            ),
            post: Conv1dReluBn::new(channels, channels, 1, 1, 0, 1, true, device),
            se: SeConnect::new(channels, 128, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let y = self.pre.forward(x.clone());
        let y = self.res2.forward(y);
        let y = self.post.forward(y);
        let y = self.se.forward(y);

        x + y
    }
}

#[derive(Module, Debug)]
pub struct Astp<B: Backend> {
    pub linear1: Conv1d<B>,
    pub linear2: Conv1d<B>,
    pub in_dim: usize,
    pub global_context_att: bool,
}

impl<B: Backend> Astp<B> {
    pub fn new(
        in_dim: usize,
        bottleneck_dim: usize,
        global_context_att: bool,
        device: &B::Device,
    ) -> Self {
        let attn_in_dim = if global_context_att {
            in_dim * 3
        } else {
            in_dim
        };

        Self {
            linear1: Conv1dConfig::new(attn_in_dim, bottleneck_dim, 1).init(device),
            linear2: Conv1dConfig::new(bottleneck_dim, in_dim, 1).init(device),
            in_dim,
            global_context_att,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, channels, num_frames] = x.dims();
        let x_in = if self.global_context_att {
            let context_mean = x
                .clone()
                .mean_dim(2)
                .expand([batch_size, channels, num_frames]);
            let context_std = (x.clone().var(2) + 1e-7)
                .sqrt()
                .expand([batch_size, channels, num_frames]);

            Tensor::cat(vec![x.clone(), context_mean, context_std], 1)
        } else {
            x.clone()
        };

        let alpha = softmax(self.linear2.forward(tanh(self.linear1.forward(x_in))), 2);
        let mean = (alpha.clone() * x.clone())
            .sum_dim(2)
            .squeeze_dims::<2>(&[2_isize]);
        let second_moment = (alpha * x.square())
            .sum_dim(2)
            .squeeze_dims::<2>(&[2_isize]);
        let var = second_moment - mean.clone().square();
        let std = var.clamp_min(1e-7).sqrt();

        Tensor::cat(vec![mean, std], 1)
    }
}

#[derive(Module, Debug)]
pub struct EcapaTdnn<B: Backend> {
    pub layer1: Conv1dReluBn<B>,
    pub layer2: SeRes2Block<B>,
    pub layer3: SeRes2Block<B>,
    pub layer4: SeRes2Block<B>,
    pub conv: Conv1d<B>,
    pub pool: Astp<B>,
    pub bn: BatchNorm<B>,
    pub linear: Linear<B>,
}

impl<B: Backend> EcapaTdnn<B> {
    pub fn new(config: &EcapaTdnnConfig, device: &B::Device) -> Self {
        Self {
            layer1: Conv1dReluBn::new(config.feat_dim, config.channels, 5, 1, 2, 1, true, device),
            layer2: SeRes2Block::new(config.channels, 3, 1, 2, 2, 8, device),
            layer3: SeRes2Block::new(config.channels, 3, 1, 3, 3, 8, device),
            layer4: SeRes2Block::new(config.channels, 3, 1, 4, 4, 8, device),
            conv: Conv1dConfig::new(config.channels * 3, config.channels * 3, 1).init(device),
            pool: Astp::new(config.channels * 3, 128, config.global_context_att, device),
            bn: BatchNormConfig::new(config.channels * 6).init(device),
            linear: LinearConfig::new(config.channels * 6, config.embed_dim).init(device),
        }
    }

    pub fn frame_level_features(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let x = x.permute([0, 2, 1]);
        let out1 = self.layer1.forward(x);
        let out2 = self.layer2.forward(out1);
        let out3 = self.layer3.forward(out2.clone());
        let out4 = self.layer4.forward(out3.clone());
        let out = Tensor::cat(vec![out2, out3, out4.clone()], 1);

        (self.conv.forward(out), out4)
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let (out, out4) = self.frame_level_features(x);
        let out = relu(out);
        let out = self.bn.forward(self.pool.forward(out));
        let out = self.linear.forward(out);

        (out4, out)
    }
}
