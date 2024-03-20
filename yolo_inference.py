from ultralytics import YOLO
import torch

print("CUDA Available: ", torch.cuda.is_available())

model = YOLO('yolov8x.pt')

# # ensure we are using the GPU
# device = "0" if torch.cuda.is_available() else "cpu"
# if device == "0":
#     torch.cuda.set_device(0)

# print("Using Device: ", device)


model.predict('input_videos/image.png', save=True)
model.predict('input_videos/input_video.mp4', save=True)