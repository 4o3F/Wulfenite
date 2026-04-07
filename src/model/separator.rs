use burn::{
    module::{Ignored, Module},
    nn::{
        BiLstm, BiLstmConfig, GroupNorm, GroupNormConfig, Linear, LinearConfig,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{Tensor, activation::tanh, backend::Backend},
};

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeakerFuseType {
    Concat,
    Additive,
    Multiply,
}

#[derive(Module, Debug)]
pub struct SpeakerFuseLayer<B: Backend> {
    pub fc: Linear<B>,
    pub fuse_type: Ignored<SpeakerFuseType>,
}

impl<B: Backend> SpeakerFuseLayer<B> {
    pub fn new(
        embed_dim: usize,
        feat_dim: usize,
        fuse_type: SpeakerFuseType,
        device: &B::Device,
    ) -> Self {
        let in_features = match fuse_type {
            SpeakerFuseType::Concat => embed_dim + feat_dim,
            SpeakerFuseType::Additive | SpeakerFuseType::Multiply => embed_dim,
        };

        Self {
            fc: LinearConfig::new(in_features, feat_dim).init(device),
            fuse_type: Ignored(fuse_type),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>, embed: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, nband, _feat_dim, num_frames] = x.dims();
        let embed_dim = embed.dims()[2];
        let embed_t = embed.expand([batch_size, nband, embed_dim, num_frames]);

        match self.fuse_type.0 {
            SpeakerFuseType::Concat => {
                let y = Tensor::cat(vec![x.clone(), embed_t], 2).swap_dims(2, 3);
                self.fc.forward(y).swap_dims(2, 3)
            }
            SpeakerFuseType::Additive => {
                let transformed = self.fc.forward(embed_t.swap_dims(2, 3)).swap_dims(2, 3);
                x + transformed
            }
            SpeakerFuseType::Multiply => {
                let transformed = self.fc.forward(embed_t.swap_dims(2, 3)).swap_dims(2, 3);
                x * transformed
            }
        }
    }
}

#[derive(Module, Debug)]
pub struct ResRnn<B: Backend> {
    pub norm: GroupNorm<B>,
    pub rnn: BiLstm<B>,
    pub proj: Linear<B>,
}

impl<B: Backend> ResRnn<B> {
    pub fn new(input_size: usize, hidden_size: usize, device: &B::Device) -> Self {
        let eps = f32::EPSILON as f64;

        Self {
            norm: GroupNormConfig::new(1, input_size)
                .with_epsilon(eps)
                .init(device),
            rnn: BiLstmConfig::new(input_size, hidden_size, true)
                .with_batch_first(true)
                .init(device),
            proj: LinearConfig::new(hidden_size * 2, input_size).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let normed = self.norm.forward(x.clone());
        let (rnn_output, _) = self.rnn.forward(normed.swap_dims(1, 2), None);
        let projected = self.proj.forward(rnn_output).swap_dims(1, 2);

        x + projected
    }
}

#[derive(Module, Debug)]
pub struct BsNet<B: Backend> {
    pub band_rnn: ResRnn<B>,
    pub band_comm: ResRnn<B>,
    pub nband: usize,
    pub feature_dim: usize,
}

impl<B: Backend> BsNet<B> {
    pub fn new(in_channel: usize, nband: usize, device: &B::Device) -> Self {
        let feature_dim = in_channel / nband;

        Self {
            band_rnn: ResRnn::new(feature_dim, feature_dim * 2, device),
            band_comm: ResRnn::new(feature_dim, feature_dim * 2, device),
            nband,
            feature_dim,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, _, seq_len] = x.dims();

        let band_output =
            self.band_rnn
                .forward(x.reshape([batch_size * self.nband, self.feature_dim, seq_len]));
        let band_output = band_output.reshape([batch_size, self.nband, self.feature_dim, seq_len]);
        let band_output = band_output.permute([0, 3, 2, 1]).reshape([
            batch_size * seq_len,
            self.feature_dim,
            self.nband,
        ]);
        let output = self.band_comm.forward(band_output);

        output
            .reshape([batch_size, seq_len, self.feature_dim, self.nband])
            .permute([0, 3, 2, 1])
            .reshape([batch_size, self.nband * self.feature_dim, seq_len])
    }
}

#[derive(Module, Debug)]
pub struct FuseSeparation<B: Backend> {
    pub speaker_fuse: SpeakerFuseLayer<B>,
    pub blocks: Vec<BsNet<B>>,
    pub nband: usize,
    pub feature_dim: usize,
}

impl<B: Backend> FuseSeparation<B> {
    pub fn new(
        nband: usize,
        num_repeat: usize,
        feature_dim: usize,
        spk_emb_dim: usize,
        spk_fuse_type: SpeakerFuseType,
        device: &B::Device,
    ) -> Self {
        let mut blocks = Vec::with_capacity(num_repeat);
        for _ in 0..num_repeat {
            blocks.push(BsNet::new(nband * feature_dim, nband, device));
        }

        Self {
            speaker_fuse: SpeakerFuseLayer::new(spk_emb_dim, feature_dim, spk_fuse_type, device),
            blocks,
            nband,
            feature_dim,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>, spk_embedding: Tensor<B, 4>) -> Tensor<B, 4> {
        let batch_size = x.dims()[0];
        let num_frames = x.dims()[3];

        let mut x = self.speaker_fuse.forward(x, spk_embedding).reshape([
            batch_size,
            self.nband * self.feature_dim,
            num_frames,
        ]);

        for block in self.blocks.iter() {
            x = block.forward(x);
        }

        x.reshape([batch_size, self.nband, self.feature_dim, num_frames])
    }
}

#[derive(Module, Debug)]
pub struct BandNormProjection<B: Backend> {
    pub norm: GroupNorm<B>,
    pub proj: Conv1d<B>,
}

impl<B: Backend> BandNormProjection<B> {
    pub fn new(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        let eps = f32::EPSILON as f64;

        Self {
            norm: GroupNormConfig::new(1, in_channels)
                .with_epsilon(eps)
                .init(device),
            proj: Conv1dConfig::new(in_channels, out_channels, 1).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.proj.forward(self.norm.forward(x))
    }
}

#[derive(Module, Debug)]
pub struct MaskHead<B: Backend> {
    pub norm: GroupNorm<B>,
    pub fc1: Conv1d<B>,
    pub fc2: Conv1d<B>,
    pub fc3: Conv1d<B>,
}

impl<B: Backend> MaskHead<B> {
    pub fn new(feature_dim: usize, band_width: usize, device: &B::Device) -> Self {
        let eps = f32::EPSILON as f64;

        Self {
            norm: GroupNormConfig::new(1, feature_dim)
                .with_epsilon(eps)
                .init(device),
            fc1: Conv1dConfig::new(feature_dim, feature_dim * 4, 1).init(device),
            fc2: Conv1dConfig::new(feature_dim * 4, feature_dim * 4, 1).init(device),
            fc3: Conv1dConfig::new(feature_dim * 4, band_width * 4, 1).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.norm.forward(x);
        let x = tanh(self.fc1.forward(x));
        let x = tanh(self.fc2.forward(x));

        self.fc3.forward(x)
    }
}
