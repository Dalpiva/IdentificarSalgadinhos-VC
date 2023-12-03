# Referencia GIT
# https://github.com/ultralytics/ultralytics

# Referencia Site
# https://docs.ultralytics.com/modes/train/#clearml


from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

results = model.train(data="config.yaml", epochs=400)
