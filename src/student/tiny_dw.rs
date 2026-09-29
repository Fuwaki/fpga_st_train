//! MobileNet-style depthwise-separable student for FPGA-friendly deployment.
//!
//! Same topology as [super::tiny::TinyStudent] (encoder -> residual trunk -> nearest
//! upsampling decoder), but every 3x3 convolution is split into depthwise 3x3 + pointwise
//! 1x1. On hardware the depthwise stage is one MAC per channel per tap; the pointwise stage
//! is a small 1x1 mixer. Both map cleanly onto a line-buffer datapath without grouped or
//! transposed convolutions beyond depthwise.

use burn::module::Module;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::prelude::*;
use burn::tensor::activation::relu;
use burn::tensor::module::{conv2d, interpolate};
use burn::tensor::ops::{InterpolateMode, InterpolateOptions, PadMode, PaddedConvOptions};

use crate::quant;
use crate::student::StudentModel;
use crate::student::tiny::Conv;

/// ReflectionPad(k/2) + depthwise Conv2d (groups = in_channels).
#[derive(Module, Debug)]
pub struct DwConv<B: Backend> {
    pub conv: Conv2d<B>,
    pad: usize,
}

impl<B: Backend> DwConv<B> {
    pub fn new(channels: usize, kernel: usize, stride: usize, device: &B::Device) -> Self {
        let conv = Conv2dConfig::new([channels, channels], [kernel, kernel])
            .with_stride([stride, stride])
            .with_groups(channels)
            .with_padding(PaddingConfig2d::Valid)
            .with_bias(true)
            .init(device);
        Self {
            conv,
            pad: kernel / 2,
        }
    }

    pub fn reflect_pad(&self) -> usize {
        self.pad
    }

    pub fn weight(&self) -> Tensor<B, 4> {
        self.conv.weight.val()
    }

    pub fn forward_with(&self, x: Tensor<B, 4>, weight: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = if self.pad > 0 {
            x.pad(
                [(self.pad, self.pad), (self.pad, self.pad)],
                PadMode::Reflect,
            )
        } else {
            x
        };
        let bias = self.conv.bias.as_ref().map(|b| b.val());
        conv2d(
            x,
            weight,
            bias,
            PaddedConvOptions::asymmetric(
                self.conv.stride,
                [0, 0],
                [0, 0],
                self.conv.dilation,
                self.conv.groups,
            ),
        )
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward_with(x, self.weight())
    }
}

/// Pointwise 1x1 Conv2d.
#[derive(Module, Debug)]
pub struct PwConv<B: Backend> {
    pub conv: Conv2d<B>,
}

impl<B: Backend> PwConv<B> {
    pub fn new(in_ch: usize, out_ch: usize, device: &B::Device) -> Self {
        let conv = Conv2dConfig::new([in_ch, out_ch], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_bias(true)
            .init(device);
        Self { conv }
    }

    pub fn weight(&self) -> Tensor<B, 4> {
        self.conv.weight.val()
    }

    pub fn forward_with(&self, x: Tensor<B, 4>, weight: Tensor<B, 4>) -> Tensor<B, 4> {
        let bias = self.conv.bias.as_ref().map(|b| b.val());
        conv2d(
            x,
            weight,
            bias,
            PaddedConvOptions::asymmetric(
                self.conv.stride,
                [0, 0],
                [0, 0],
                self.conv.dilation,
                self.conv.groups,
            ),
        )
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward_with(x, self.weight())
    }
}

/// Depthwise 3x3 followed by pointwise 1x1 (MobileNet separable block).
#[derive(Module, Debug)]
pub struct SepConv<B: Backend> {
    pub dw: DwConv<B>,
    pub pw: PwConv<B>,
}

impl<B: Backend> SepConv<B> {
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            dw: DwConv::new(in_ch, kernel, stride, device),
            pw: PwConv::new(in_ch, out_ch, device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward_inner(x, true)
    }

    /// Depthwise + ReLU + pointwise. `out_relu` is false for the residual addend and the RGB head,
    /// matching the dense student: those outputs must stay signed.
    fn forward_inner(&self, x: Tensor<B, 4>, out_relu: bool) -> Tensor<B, 4> {
        let y = self.pw.forward(relu(self.dw.forward(x)));
        if out_relu { relu(y) } else { y }
    }

    fn forward_quant<F: FnMut(usize, &Tensor<B, 4>)>(
        &self,
        x: Tensor<B, 4>,
        dw_point: usize,
        pw_point: usize,
        scheme: Option<&quant::QuantScheme>,
        observe: &mut F,
        wq_dw: Tensor<B, 4>,
        wq_pw: Tensor<B, 4>,
        out_relu: bool,
    ) -> Tensor<B, 4> {
        let h = qa(x, scheme, dw_point, observe);
        let h = relu(self.dw.forward_with(h, wq_dw));
        let h = qa(h, scheme, pw_point, observe);
        let y = self.pw.forward_with(h, wq_pw);
        if out_relu { relu(y) } else { y }
    }
}

fn qa<B: Backend, F: FnMut(usize, &Tensor<B, 4>)>(
    x: Tensor<B, 4>,
    scheme: Option<&quant::QuantScheme>,
    point: usize,
    observe: &mut F,
) -> Tensor<B, 4> {
    observe(point, &x);
    match scheme {
        Some(scheme) => quant::fake_quant_act(x, scheme.acts[point], scheme.act_max()),
        None => x,
    }
}

fn wq_dw<B: Backend>(conv: &DwConv<B>, scheme: Option<&quant::QuantScheme>) -> Tensor<B, 4> {
    let weight = conv.weight();
    match scheme {
        Some(scheme) => quant::fake_quant_weight(weight, scheme.weight_max()),
        None => weight,
    }
}

fn wq_pw<B: Backend>(conv: &PwConv<B>, scheme: Option<&quant::QuantScheme>) -> Tensor<B, 4> {
    let weight = conv.weight();
    match scheme {
        Some(scheme) => quant::fake_quant_weight(weight, scheme.weight_max()),
        None => weight,
    }
}

/// Two separable 3x3 blocks with a residual connection.
#[derive(Module, Debug)]
pub struct ResBlockDw<B: Backend> {
    pub s1: SepConv<B>,
    pub s2: SepConv<B>,
}

impl<B: Backend> ResBlockDw<B> {
    fn new(channels: usize, device: &B::Device) -> Self {
        Self {
            s1: SepConv::new(channels, channels, 3, 1, device),
            s2: SepConv::new(channels, channels, 3, 1, device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mid = self.s1.forward(x.clone());
        let inner = self.s2.forward_inner(mid, false);
        x + inner
    }
}

fn upsample2x<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [_, _, h, w] = x.dims();
    interpolate(
        x,
        [h * 2, w * 2],
        InterpolateOptions::new(InterpolateMode::Nearest),
    )
}

/// Depthwise-separable variant of [super::tiny::TinyStudent].
#[derive(Module, Debug)]
pub struct TinyDwStudent<B: Backend> {
    /// Dense 3x3 stem. A depthwise stem cannot mix RGB inside the kernel.
    pub stem: Conv<B>,
    pub enc: Vec<SepConv<B>>,
    pub res: Vec<ResBlockDw<B>>,
    pub up3: Vec<SepConv<B>>,
    pub up1: Vec<PwConv<B>>,
    pub head: SepConv<B>,
}

impl<B: Backend> TinyDwStudent<B> {
    pub fn new(
        base: usize,
        res_blocks: usize,
        mid: usize,
        fine: usize,
        device: &B::Device,
    ) -> Self {
        let c1 = base;
        let c2 = base * 2;
        let stem = Conv::new(3, c1, 3, 1, true, device);
        let enc = vec![
            SepConv::new(c1, c2, 3, 2, device),
            SepConv::new(c2, c2, 3, 2, device),
        ];
        let res = (0..res_blocks)
            .map(|_| ResBlockDw::new(c2, device))
            .collect();
        let up3 = vec![
            SepConv::new(c2, mid, 3, 1, device),
            SepConv::new(mid, fine, 3, 1, device),
        ];
        let up1 = vec![
            PwConv::new(mid, mid, device),
            PwConv::new(fine, fine, device),
        ];
        let head = SepConv::new(fine, 3, 3, 1, device);
        Self {
            stem,
            enc,
            res,
            up3,
            up1,
            head,
        }
    }

    /// One quantised point per convolution input (dw + pw for each SepConv, pw for up1).
    pub fn quant_point_count(&self) -> usize {
        1 + 2 * self.enc.len() + 4 * self.res.len() + 2 * self.up3.len() + self.up1.len() + 2
    }

    pub fn quant_layout(&self) -> Vec<String> {
        let mut names = vec!["image".to_owned()];
        for i in 0..self.enc.len() {
            names.push(format!("enc.{i}.dw.in"));
            names.push(format!("enc.{i}.pw.in"));
        }
        for (r, _) in self.res.iter().enumerate() {
            names.push(format!("res.{r}.s1.dw.in"));
            names.push(format!("res.{r}.s1.pw.in"));
            names.push(format!("res.{r}.s2.dw.in"));
            names.push(format!("res.{r}.s2.pw.in"));
        }
        names.push("up3.0.dw.in".to_owned());
        names.push("up3.0.pw.in".to_owned());
        names.push("up1.0.in".to_owned());
        names.push("up3.1.dw.in".to_owned());
        names.push("up3.1.pw.in".to_owned());
        names.push("up1.1.in".to_owned());
        names.push("head.dw.in".to_owned());
        names.push("head.pw.in".to_owned());
        names
    }

    pub fn forward_core<F: FnMut(usize, &Tensor<B, 4>)>(
        &self,
        content: Tensor<B, 4>,
        scheme: Option<&quant::QuantScheme>,
        mut observe: F,
    ) -> Tensor<B, 4> {
        let mut point = 0usize;
        let mut h = qa(content, scheme, point, &mut observe);
        point += 1;
        let stem_w = match scheme {
            Some(scheme) => quant::fake_quant_weight(self.stem.weight(), scheme.weight_max()),
            None => self.stem.weight(),
        };
        h = relu(self.stem.forward_with(h, stem_w));

        for block in &self.enc {
            h = block.forward_quant(
                h,
                point,
                point + 1,
                scheme,
                &mut observe,
                wq_dw(&block.dw, scheme),
                wq_pw(&block.pw, scheme),
                true,
            );
            point += 2;
        }

        let trunk_base = point;
        for (r, block) in self.res.iter().enumerate() {
            let p = trunk_base + 4 * r;
            let input = qa(h, scheme, p, &mut observe);
            let activated = relu(
                block
                    .s1
                    .dw
                    .forward_with(input.clone(), wq_dw(&block.s1.dw, scheme)),
            );
            let activated = qa(activated, scheme, p + 1, &mut observe);
            let mid = relu(
                block
                    .s1
                    .pw
                    .forward_with(activated, wq_pw(&block.s1.pw, scheme)),
            );
            let activated = qa(mid, scheme, p + 2, &mut observe);
            let activated = relu(
                block
                    .s2
                    .dw
                    .forward_with(activated, wq_dw(&block.s2.dw, scheme)),
            );
            let activated = qa(activated, scheme, p + 3, &mut observe);
            let inner = block
                .s2
                .pw
                .forward_with(activated, wq_pw(&block.s2.pw, scheme));
            let out_point = p + 4;
            h = match scheme {
                Some(scheme) => {
                    let out = scheme.acts[out_point];
                    let qmax = scheme.act_max();
                    quant::fake_quant_act(input, out, qmax)
                        + quant::fake_quant_act(inner, out, qmax)
                }
                None => input + inner,
            };
        }

        let dec_base = trunk_base + 4 * self.res.len();
        h = upsample2x(h);

        h = qa(h, scheme, dec_base, &mut observe);
        h = self.up3[0].forward_quant(
            h,
            dec_base,
            dec_base + 1,
            scheme,
            &mut observe,
            wq_dw(&self.up3[0].dw, scheme),
            wq_pw(&self.up3[0].pw, scheme),
            true,
        );
        h = qa(h, scheme, dec_base + 2, &mut observe);
        h = relu(self.up1[0].forward_with(h, wq_pw(&self.up1[0], scheme)));

        h = upsample2x(h);

        h = qa(h, scheme, dec_base + 3, &mut observe);
        h = self.up3[1].forward_quant(
            h,
            dec_base + 3,
            dec_base + 4,
            scheme,
            &mut observe,
            wq_dw(&self.up3[1].dw, scheme),
            wq_pw(&self.up3[1].pw, scheme),
            true,
        );
        h = qa(h, scheme, dec_base + 5, &mut observe);
        h = relu(self.up1[1].forward_with(h, wq_pw(&self.up1[1], scheme)));

        h = qa(h, scheme, dec_base + 6, &mut observe);
        let image = self.head.forward_quant(
            h,
            dec_base + 6,
            dec_base + 7,
            scheme,
            &mut observe,
            wq_dw(&self.head.dw, scheme),
            wq_pw(&self.head.pw, scheme),
            false,
        );
        match scheme {
            Some(_) => quant::fake_quant_output(image),
            None => image,
        }
    }

    fn intermediates(&self, content: Tensor<B, 4>) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        let mut h = relu(self.stem.forward(content));
        for block in &self.enc {
            h = block.forward(h);
        }
        for block in &self.res {
            h = block.forward(h);
        }
        let trunk = h.clone();

        h = upsample2x(h);
        h = self.up3[0].forward(h);
        h = relu(self.up1[0].forward(h));
        let half = h.clone();

        h = upsample2x(h);
        h = self.up3[1].forward(h);
        h = relu(self.up1[1].forward(h));
        let full = h.clone();

        (self.head.forward_inner(h, false), vec![trunk, half, full])
    }
}

impl<B: Backend> StudentModel<B> for TinyDwStudent<B> {
    fn stylize(&self, content: Tensor<B, 4>) -> Tensor<B, 4> {
        self.intermediates(content).0
    }

    fn intermediate(&self, content: Tensor<B, 4>) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        self.intermediates(content)
    }

    fn forward_quantised(
        &self,
        content: Tensor<B, 4>,
        scheme: &quant::QuantScheme,
    ) -> Tensor<B, 4> {
        self.forward_core(content, Some(scheme), |_, _| {})
    }

    fn quant_points(&self) -> Vec<String> {
        self.quant_layout()
    }

    fn observe_activations(&self, content: Tensor<B, 4>, observe: &mut dyn FnMut(usize, f32)) {
        self.forward_core(content, None, |point, x| {
            observe(point, x.clone().abs().max().into_scalar().to_f64() as f32);
        });
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn tiny_dw_has_fewer_params_than_tiny() {
        fn conv3(in_ch: usize, out_ch: usize) -> usize {
            out_ch * in_ch * 9 + out_ch
        }
        fn sep3(in_ch: usize, out_ch: usize) -> usize {
            in_ch * 9 + in_ch + out_ch * in_ch + out_ch
        }
        fn pw1(in_ch: usize, out_ch: usize) -> usize {
            out_ch * in_ch + out_ch
        }

        let base = 32usize;
        let res_blocks = 4usize;
        let mid = 32usize;
        let fine = 16usize;
        let c2 = base * 2;

        let tiny = conv3(3, base)
            + conv3(base, c2)
            + conv3(c2, c2)
            + res_blocks * (conv3(c2, c2) + conv3(c2, c2))
            + conv3(c2, mid)
            + pw1(mid, mid)
            + conv3(mid, fine)
            + pw1(fine, fine)
            + conv3(fine, 3);

        let tiny_dw = conv3(3, base)
            + sep3(base, c2)
            + sep3(c2, c2)
            + res_blocks * (sep3(c2, c2) + sep3(c2, c2))
            + sep3(c2, mid)
            + pw1(mid, mid)
            + sep3(mid, fine)
            + pw1(fine, fine)
            + sep3(fine, 3);

        assert_eq!(tiny, 376_595);
        assert_eq!(tiny_dw, 51_667);
        assert!(tiny_dw < tiny);
    }
}

#[cfg(all(test, feature = "wgpu"))]
mod forward_sanity_gpu {
    use super::*;
    use crate::backend::{InferBackend, select_device};

    #[test]
    fn tiny_dw_forward_is_not_constant_on_gpu() {
        let device = select_device("d1");
        let model = TinyDwStudent::<InferBackend>::new(32, 4, 32, 16, &device);
        let n = 1 * 3 * 64 * 64;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i * 37 % 101) as f32 / 101.0) - 0.5)
            .collect();
        let x = Tensor::<InferBackend, 4>::from_data(
            burn::tensor::TensorData::new(data, [1, 3, 64, 64]),
            &device,
        );
        let out = model.stylize(x).into_data().to_vec::<f32>().unwrap();
        let mn = out.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = out.iter().sum::<f32>() / out.len() as f32;
        eprintln!(
            "gpu d1 out mean={mean:.6} min={mn:.6} max={mx:.6} span={:.6}",
            mx - mn
        );
        assert!(mx - mn > 0.01, "GPU output is constant: min={mn} max={mx}");
    }

    #[test]
    fn tiny_dw_forward_large_resolution_on_gpu() {
        let device = select_device("d1");
        let model = TinyDwStudent::<InferBackend>::new(32, 4, 32, 16, &device);
        for (h, w) in [(192, 192), (288, 384), (512, 512)] {
            let n = 1 * 3 * h * w;
            let data: Vec<f32> = (0..n)
                .map(|i| ((i * 37 % 101) as f32 / 101.0) - 0.5)
                .collect();
            let x = Tensor::<InferBackend, 4>::from_data(
                burn::tensor::TensorData::new(data, [1, 3, h, w]),
                &device,
            );
            let out = model.stylize(x).into_data().to_vec::<f32>().unwrap();
            let mn = out.iter().cloned().fold(f32::INFINITY, f32::min);
            let mx = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean = out.iter().sum::<f32>() / out.len() as f32;
            eprintln!(
                "gpu {h}x{w} mean={mean:.6} min={mn:.6} max={mx:.6} span={:.6}",
                mx - mn
            );
            assert!(mx - mn > 0.01, "{h}x{w} output is constant");
        }
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod forward_sanity_cpu {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn tiny_dw_forward_is_not_constant_on_cpu() {
        let device = Default::default();
        let model = TinyDwStudent::<NdArray>::new(32, 4, 32, 16, &device);
        let n = 1 * 3 * 64 * 64;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i * 37 % 101) as f32 / 101.0) - 0.5)
            .collect();
        let x = Tensor::<NdArray, 4>::from_data(
            burn::tensor::TensorData::new(data, [1, 3, 64, 64]),
            &device,
        );
        let out = model.stylize(x).into_data().to_vec::<f32>().unwrap();
        let mn = out.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("cpu out min={mn} max={mx}");
        assert!(mx - mn > 0.01, "output is constant: min={mn} max={mx}");
    }
}
