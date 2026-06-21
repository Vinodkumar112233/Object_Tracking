# TODO: Fine-tune YOLOv8 for custom vehicle classes

- [ ] Provide dataset folder structure (train/val/test -> images/ + labels/)
- [ ] Provide example `data.yaml` format for custom classes
- [ ] Create `train_custom_yolov8.py` using Ultralytics YOLO API (transfer learning from `yolov8n.pt`, save best weights)
- [ ] Create `infer_custom_yolov8.py` that loads resulting `best.pt` for inference
- [ ] Provide validation instructions (mAP) after training

