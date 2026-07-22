"""Plain YOLO pretraining/baseline run — no dual-domain code involved.

Produces:
  - the checkpoint DualDomainTrainer expects as `weights=` (the model is assumed already
    pretrained on the source domain before dual-domain adaptation starts), and
  - in-domain / cross-domain baseline numbers to compare MMD results against.

Saves to its own project dir (runs/pretrain/<name>) so it doesn't collide with
dual-domain training runs. Point `data=` at whichever single-domain yaml you want a
baseline for (see examples/source_data.yaml).
"""

from ultralytics import YOLO

model = YOLO("yolov10n.yaml")
model.train(
    data="examples/source_data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    project="runs/pretrain",
    name="source",
)
