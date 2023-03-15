import os
import argparse
import time
import json
import numpy as np

import onnx
from onnxsim import simplify
import onnxruntime as ort
import torch

from models import SynthesizerTrn
from utils import load_checkpoint
#from text.symbols import symbols


def get_hparams_from_file(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)
    hparams = HParams(**config)
    return hparams


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--convert_pth", required=True)
    return parser.parse_args()


def inspect_onnx(session):
    print("inputs")
    for i in session.get_inputs():
        print("name:{}\tshape:{}\tdtype:{}".format(i.name, i.shape, i.type))
    print("outputs")
    for i in session.get_outputs():
        print("name:{}\tshape:{}\tdtype:{}".format(i.name, i.shape, i.type))


def benchmark(session, hps, conf):
    hop_length = hps.data.hop_length
    delay_frames = conf.vc_conf.delay_flames
    overlap_length = conf.vc_conf.overlap
    dispose_stft_specs = conf.vc_conf.dispose_stft_specs
    dispose_conv1d_specs = conf.vc_conf.dispose_conv1d_specs
    dispose_specs =  dispose_stft_specs * 2 + dispose_conv1d_specs * 2
    dispose_length = dispose_specs * hop_length
    fixed_length = (delay_frames + dispose_length + overlap_length) // hop_length - dispose_stft_specs * 2
    upsample_scales = hps.model.upsample_rates
    prod_upsample_scales = np.cumprod(upsample_scales)
    dummy_specs = torch.rand(1, 257, fixed_length).numpy()
    dummy_lengths = torch.LongTensor([fixed_length]).numpy()
    dummy_sin = torch.rand(1, 1, fixed_length * hop_length).numpy()
    dummy_d0  = torch.rand(1, 1, fixed_length * prod_upsample_scales[0]).numpy()
    dummy_d1  = torch.rand(1, 1, fixed_length * prod_upsample_scales[1]).numpy()
    dummy_d2  = torch.rand(1, 1, fixed_length * prod_upsample_scales[2]).numpy()
    dummy_d3  = torch.rand(1, 1, fixed_length * prod_upsample_scales[3]).numpy()
    dummy_sid_src = torch.LongTensor([0]).numpy()
    dummy_sid_tgt = torch.LongTensor([1]).numpy()

    use_time_list = []
    for i in range(30):
        start = time.time()
        output = session.run(
            ["audio"],
            {
                "specs": dummy_specs,
                "lengths": dummy_lengths,
                "sin": dummy_sin,
                "d0": dummy_d0,
                "d1": dummy_d1,
                "d2": dummy_d2,
                "d3": dummy_d3,
                "sid_src": dummy_sid_src,
                "sid_tgt": dummy_sid_tgt
            }
        )
        use_time = time.time() - start
        use_time_list.append(use_time)
        #print("use time:{}".format(use_time))
    use_time_list = use_time_list[5:]
    mean_use_time = sum(use_time_list) / len(use_time_list)
    print(f"mean_use_time:{mean_use_time}")


class OnnxSynthesizerTrn(SynthesizerTrn):
    def forward(self, y, lengths, sin, d0, d1, d2, d3, sid_src, sid_tgt):
        return self.voice_conversion(y, lengths, sin, (d0, d1, d2, d3), sid_src, sid_tgt)

    def voice_conversion(self, y, lengths, sin, d, sid_src, sid_tgt):
        assert self.n_speakers > 0, "n_speakers have to be larger than 0."
        g_src = self.emb_g(sid_src).unsqueeze(-1)
        g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
        z, _, _, y_mask = self.enc_q(y, lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(sin, z_hat * y_mask, d, sid=g_tgt)
        return o_hat


def main(args):
    hps = get_hparams_from_file(args.config_file)
    conf = get_hparams_from_file("debug_test_15.conf")
    net_g = OnnxSynthesizerTrn(
        spec_channels = hps.data.filter_length // 2 + 1,
        segment_size = hps.train.segment_size // hps.data.hop_length,
        inter_channels = hps.model.inter_channels,
        hidden_channels = hps.model.hidden_channels,
        upsample_rates = hps.model.upsample_rates,
        upsample_initial_channel = hps.model.upsample_initial_channel,
        upsample_kernel_sizes = hps.model.upsample_kernel_sizes,
        n_flow = hps.model.n_flow,
        dec_out_channels=1,
        dec_kernel_size=7,
        n_speakers = hps.data.n_speakers,
        gin_channels = hps.model.gin_channels,
        requires_grad_pe = hps.requires_grad.pe,
        requires_grad_flow = hps.requires_grad.flow,
        requires_grad_text_enc = hps.requires_grad.text_enc,
        requires_grad_dec = hps.requires_grad.dec,
        requires_grad_emb_g = hps.requires_grad.emb_g
    )
    for i in net_g.parameters():
        i.requires_grad = False
    _ = net_g.eval()
    _ = load_checkpoint(args.convert_pth, net_g, generator=True, optimizer=None)
    print("Model data loading succeeded.\nConverting start.")

    # Convert to ONNX
    hop_length = hps.data.hop_length
    delay_frames = conf.vc_conf.delay_flames
    overlap_length = conf.vc_conf.overlap
    dispose_stft_specs = conf.vc_conf.dispose_stft_specs
    dispose_conv1d_specs = conf.vc_conf.dispose_conv1d_specs
    dispose_specs =  dispose_stft_specs * 2 + dispose_conv1d_specs * 2
    dispose_length = dispose_specs * hop_length
    fixed_length = (delay_frames + dispose_length + overlap_length) // hop_length - dispose_stft_specs * 2
    upsample_scales = hps.model.upsample_rates
    prod_upsample_scales = np.cumprod(upsample_scales)
    dummy_specs = torch.rand(1, 257, fixed_length)
    dummy_lengths = torch.LongTensor([fixed_length])
    dummy_sin = torch.rand(1, 1, fixed_length * hop_length)
    dummy_d0  = torch.rand(1, 1, fixed_length * prod_upsample_scales[0])
    dummy_d1  = torch.rand(1, 1, fixed_length * prod_upsample_scales[1])
    dummy_d2  = torch.rand(1, 1, fixed_length * prod_upsample_scales[2])
    dummy_d3  = torch.rand(1, 1, fixed_length * prod_upsample_scales[3])
    dummy_sid_src = torch.LongTensor([0])
    dummy_sid_tgt = torch.LongTensor([1])

    dirname = os.path.dirname(args.convert_pth)
    filenames = os.path.splitext(os.path.basename(args.convert_pth))
    onnx_file = os.path.join(dirname, filenames[0] + ".onnx")

    torch.onnx.export(
        net_g,
        (dummy_specs, dummy_lengths, dummy_sin, dummy_d0, dummy_d1, dummy_d2, dummy_d3, dummy_sid_src, dummy_sid_tgt),
        #(dummy_specs, dummy_lengths, dummy_f0, dummy_sid_src, dummy_sid_tgt),
        onnx_file,
        do_constant_folding=False,
        #opset_version=13,
        opset_version=17,
        verbose=False,
        input_names=["specs", "lengths", "sin", "d0", "d1", "d2", "d3", "sid_src", "sid_tgt"],
        output_names=["audio"])
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    print("Done\n")

    print("vits onnx benchmark")
    ort_session_cpu = ort.InferenceSession(
        onnx_file,
        providers=["CPUExecutionProvider"])
    inspect_onnx(ort_session_cpu)
    print("ONNX CPU")
    benchmark(ort_session_cpu, hps, conf)


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
