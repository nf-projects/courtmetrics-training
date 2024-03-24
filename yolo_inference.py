from ultralytics import YOLO
import torch

print("CUDA Available: ", torch.cuda.is_available())

# to track people
# model = YOLO('yolov8x.pt')

# to track the tennis ball
# model = YOLO('models/yolov5_best.pt')
model = YOLO('training/keypoints_model.pth')

# # ensure we are using the GPU
# device = "0" if torch.cuda.is_available() else "cpu"
# if device == "0":
#     torch.cuda.set_device(0)

# print("Using Device: ", device)


model.predict('input_videos/image.png', conf=0.1, save=True)
model.predict('input_videos/input_video.mp4', conf=0.1, save=True)