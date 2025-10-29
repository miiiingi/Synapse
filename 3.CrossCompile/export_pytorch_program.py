import tvm
import torch
from torch._export.converter import TS2EPConverter
from exported_program_translator import from_exported_program


@torch.no_grad()
def export_model(model_path, example_inputs):
    model = torch.jit.load(model_path)
    model.eval()
    converter = TS2EPConverter(model, example_inputs, {})
    exported_program = converter.convert()
    return exported_program


# 더미 입력 정의
input_shape = (1, 3, 192, 640)
input_data = (torch.randn(input_shape),)
target = "llvm -mtriple=aarch64-linux-gnu"  # or "cuda" for GPU

# TorchScript 모델 로드, exported program으로 변환
encoder_program = export_model("model/mono_640x192_encoder.pt", input_data)
encoder_relay_mod = from_exported_program(encoder_program)

with tvm.transform.PassContext(opt_level=3):
    lib = tvm.relay.build(encoder_relay_mod, target=target)


encoder_lib_path = "encoder_deploy_tvm.so"
lib.export_library(encoder_lib_path, cc="aarch64-linux-gnu-g++")
print(f"✅ TVM Encoder shared library exported: {encoder_lib_path}")

decoder_program = export_model("model/mono_640x192_decoder.pt", input_data)
decoder_relay_mod = from_exported_program(decoder_program)

with tvm.transform.PassContext(opt_level=3):
    lib = tvm.relay.build(decoder_relay_mod, target=target)


decoder_lib_path = "decoder_deploy_tvm.so"
lib.export_library(decoder_lib_path, cc="aarch64-linux-gnu-g++")
print(f"✅ TVM Decoder shared library exported: {decoder_lib_path}")

