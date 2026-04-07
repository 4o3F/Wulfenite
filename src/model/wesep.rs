use burn::{
    module::{Ignored, Module},
    tensor::{Tensor, activation::sigmoid, backend::Backend},
};

use crate::model::{
    ecapa::{EcapaTdnn, EcapaTdnnConfig},
    separator::{BandNormProjection, FuseSeparation, MaskHead, SpeakerFuseType},
};

#[derive(Debug)]
pub struct ComplexSpectrogram<B: Backend> {
    pub real: Tensor<B, 3>,
    pub imag: Tensor<B, 3>,
}

#[derive(Clone, Debug)]
pub struct WeSepBsrnnConfig {
    pub sample_rate: usize,
    pub win: usize,
    pub stride: usize,
    pub feature_dim: usize,
    pub spk_emb_dim: usize,
    pub speaker_channels: usize,
    pub speaker_feat_dim: usize,
    pub speaker_embed_dim: usize,
    pub num_repeat: usize,
    pub band_widths: Vec<usize>,
}

impl Default for WeSepBsrnnConfig {
    fn default() -> Self {
        let sample_rate = 16_000;
        let win = 512;
        let enc_dim = win / 2 + 1;

        Self {
            sample_rate,
            win,
            stride: 128,
            feature_dim: 128,
            spk_emb_dim: 192,
            speaker_channels: 512,
            speaker_feat_dim: 80,
            speaker_embed_dim: 192,
            num_repeat: 6,
            band_widths: Self::default_band_widths(sample_rate, enc_dim),
        }
    }
}

impl WeSepBsrnnConfig {
    pub fn enc_dim(&self) -> usize {
        self.win / 2 + 1
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> WeSepBsrnn<B> {
        assert_eq!(
            self.speaker_embed_dim, self.spk_emb_dim,
            "speaker_embed_dim must match spk_emb_dim"
        );
        assert_eq!(
            self.band_widths.iter().sum::<usize>(),
            self.enc_dim(),
            "band widths must sum to enc_dim"
        );

        let spk_model = EcapaTdnnConfig {
            channels: self.speaker_channels,
            feat_dim: self.speaker_feat_dim,
            embed_dim: self.speaker_embed_dim,
            global_context_att: true,
        }
        .init(device);

        let mut bn = Vec::with_capacity(self.band_widths.len());
        for &band_width in self.band_widths.iter() {
            bn.push(BandNormProjection::new(
                band_width * 2,
                self.feature_dim,
                device,
            ));
        }

        let separator = FuseSeparation::new(
            self.band_widths.len(),
            self.num_repeat,
            self.feature_dim,
            self.spk_emb_dim,
            SpeakerFuseType::Multiply,
            device,
        );

        let mut mask = Vec::with_capacity(self.band_widths.len());
        for &band_width in self.band_widths.iter() {
            mask.push(MaskHead::new(self.feature_dim, band_width, device));
        }

        WeSepBsrnn {
            sr: self.sample_rate,
            win: self.win,
            stride: self.stride,
            enc_dim: self.enc_dim(),
            feature_dim: self.feature_dim,
            spk_emb_dim: self.spk_emb_dim,
            nband: self.band_widths.len(),
            band_width: Ignored(self.band_widths.clone()),
            spk_model,
            bn,
            separator,
            mask,
        }
    }

    pub fn default_band_widths(sample_rate: usize, enc_dim: usize) -> Vec<usize> {
        let bandwidth_100 = (100.0 / (sample_rate as f64 / 2.0) * enc_dim as f64).floor() as usize;
        let bandwidth_200 = (200.0 / (sample_rate as f64 / 2.0) * enc_dim as f64).floor() as usize;
        let bandwidth_500 = (500.0 / (sample_rate as f64 / 2.0) * enc_dim as f64).floor() as usize;
        let bandwidth_2k = (2000.0 / (sample_rate as f64 / 2.0) * enc_dim as f64).floor() as usize;

        let mut band_widths = vec![bandwidth_100; 15];
        band_widths.extend(vec![bandwidth_200; 10]);
        band_widths.extend(vec![bandwidth_500; 5]);
        band_widths.push(bandwidth_2k);
        band_widths.push(enc_dim - band_widths.iter().sum::<usize>());
        band_widths
    }
}

#[derive(Module, Debug)]
pub struct WeSepBsrnn<B: Backend> {
    pub sr: usize,
    pub win: usize,
    pub stride: usize,
    pub enc_dim: usize,
    pub feature_dim: usize,
    pub spk_emb_dim: usize,
    pub nband: usize,
    pub band_width: Ignored<Vec<usize>>,
    pub spk_model: EcapaTdnn<B>,
    pub bn: Vec<BandNormProjection<B>>,
    pub separator: FuseSeparation<B>,
    pub mask: Vec<MaskHead<B>>,
}

impl<B: Backend> WeSepBsrnn<B> {
    pub fn forward(
        &self,
        mixture_real: Tensor<B, 3>,
        mixture_imag: Tensor<B, 3>,
        enrollment_feat: Tensor<B, 3>,
    ) -> ComplexSpectrogram<B> {
        let [batch_size, enc_dim, num_frames] = mixture_real.dims();
        assert_eq!(
            mixture_imag.dims(),
            [batch_size, enc_dim, num_frames],
            "mixture_real and mixture_imag must have the same shape"
        );
        assert_eq!(
            enc_dim, self.enc_dim,
            "expected encoder dimension {}, got {}",
            self.enc_dim, enc_dim
        );
        assert_eq!(
            enrollment_feat.dims()[2],
            self.spk_model.layer1.conv.weight.dims()[1],
            "expected enrollment features with {} mel bins",
            self.spk_model.layer1.conv.weight.dims()[1]
        );

        let mut subband_features = Vec::with_capacity(self.nband);
        let mut subband_mix_real = Vec::with_capacity(self.nband);
        let mut subband_mix_imag = Vec::with_capacity(self.nband);
        let mut band_idx = 0;

        for (&band_width, bn) in self.band_width.0.iter().zip(self.bn.iter()) {
            let real_band = mixture_real.clone().narrow(1, band_idx, band_width);
            let imag_band = mixture_imag.clone().narrow(1, band_idx, band_width);
            let band = Tensor::stack::<4>(vec![real_band.clone(), imag_band.clone()], 1).reshape([
                batch_size,
                band_width * 2,
                num_frames,
            ]);

            subband_features.push(bn.forward(band));
            subband_mix_real.push(real_band);
            subband_mix_imag.push(imag_band);
            band_idx += band_width;
        }

        let subband_feature = Tensor::stack::<4>(subband_features, 1);
        let (_, spk_embedding) = self.spk_model.forward(enrollment_feat);
        let spk_embedding = spk_embedding.unsqueeze_dims::<4>(&[1usize, 3usize]);
        let sep_output = self.separator.forward(subband_feature, spk_embedding);

        let mut est_real_bands = Vec::with_capacity(self.nband);
        let mut est_imag_bands = Vec::with_capacity(self.nband);

        for idx in 0..self.nband {
            let band_width = self.band_width.0[idx];
            let band_feature = sep_output
                .clone()
                .narrow(1, idx, 1)
                .squeeze_dims::<3>(&[1_isize]);
            let band_out = self.mask[idx]
                .forward(band_feature)
                .reshape([batch_size, 2, 2, band_width, num_frames]);
            let gate_a = band_out
                .clone()
                .narrow(1, 0, 1)
                .squeeze_dims::<4>(&[1_isize]);
            let gate_b = band_out.narrow(1, 1, 1).squeeze_dims::<4>(&[1_isize]);
            let mask = gate_a * sigmoid(gate_b);
            let mask_real = mask.clone().narrow(1, 0, 1).squeeze_dims::<3>(&[1_isize]);
            let mask_imag = mask.narrow(1, 1, 1).squeeze_dims::<3>(&[1_isize]);

            let est_real = subband_mix_real[idx].clone() * mask_real.clone()
                - subband_mix_imag[idx].clone() * mask_imag.clone();
            let est_imag = subband_mix_real[idx].clone() * mask_imag
                + subband_mix_imag[idx].clone() * mask_real;

            est_real_bands.push(est_real);
            est_imag_bands.push(est_imag);
        }

        ComplexSpectrogram {
            real: Tensor::cat(est_real_bands, 1),
            imag: Tensor::cat(est_imag_bands, 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{backend::NdArray, tensor::Distribution};

    type TestBackend = NdArray<f32>;

    #[test]
    fn default_band_widths_cover_encoder_dimension() {
        let config = WeSepBsrnnConfig::default();
        assert_eq!(config.band_widths.iter().sum::<usize>(), config.enc_dim());
    }

    #[test]
    fn small_model_preserves_spectrogram_shape() {
        let device = Default::default();
        let config = WeSepBsrnnConfig {
            sample_rate: 16_000,
            win: 16,
            stride: 4,
            feature_dim: 8,
            spk_emb_dim: 16,
            speaker_channels: 32,
            speaker_feat_dim: 80,
            speaker_embed_dim: 16,
            num_repeat: 2,
            band_widths: vec![2, 3, 4],
        };
        let model = config.init::<TestBackend>(&device);

        let mixture_real = Tensor::<TestBackend, 3>::random(
            [2, config.enc_dim(), 6],
            Distribution::Default,
            &device,
        );
        let mixture_imag = Tensor::<TestBackend, 3>::random(
            [2, config.enc_dim(), 6],
            Distribution::Default,
            &device,
        );
        let enrollment_feat =
            Tensor::<TestBackend, 3>::random([2, 10, 80], Distribution::Default, &device);

        let output = model.forward(mixture_real, mixture_imag, enrollment_feat);

        assert_eq!(output.real.dims(), [2, config.enc_dim(), 6]);
        assert_eq!(output.imag.dims(), [2, config.enc_dim(), 6]);
    }
}
