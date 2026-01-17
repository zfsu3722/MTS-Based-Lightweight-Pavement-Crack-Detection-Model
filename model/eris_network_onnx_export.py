import torch
import eris_network_deploy


def export_onnx(
    onnx_path="D:\\zhangsource\\jetson\\onnx_test\\eris_block_deploy.onnx",
    use_fp16=False
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = eris_network_deploy.ErisBlock().to(device).eval()

    if use_fp16:
        model = model.half()


    dummy_input = torch.randn(
        1, 1, 256, 256,
        device=device,
        dtype=torch.float16 if use_fp16 else torch.float32
    )

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["x"],
        output_names=["x"],
        dynamic_axes=None
    )

    print(f"ONNX exported to: {onnx_path}")


export_onnx()
