import torch

path = "/nfs/cold_project/leixiaoning/lxn/20190702/tts_projects/Tacotron2-chn/vocoder/ParallelWaveGAN/egs/didi/voc1/save_pth_model/pgan.pth"

model = torch.load("save_pth_model/pgan.pth")

torch.onnx.export(model, d, "./pgan.onnx")
