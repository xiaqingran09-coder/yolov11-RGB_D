from ultralytics import YOLO


def train_model():
    model = YOLO("yolo11s.pt")

    results = model.train(
        data="data.yaml",
        epochs=300,
        imgsz=1280,
        batch=16,
        project="cube_detect",
        name="yellow_block_v1",
        device="0"
    )

    metrics = model.val()
    print(f"mAP50: {metrics.box.map50}")


if __name__ == '__main__':
    train_model()