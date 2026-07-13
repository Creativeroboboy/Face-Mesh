import os
import shutil
from pathlib import Path

from ultralytics import YOLO


def ensure_dataset(root: Path) -> Path:
    """Create a simple YOLO-style dataset from the available ID images."""
    dataset_dir = root / "my_dataset"
    train_images = dataset_dir / "images" / "train"
    train_labels = dataset_dir / "labels" / "train"
    val_images = dataset_dir / "images" / "val"
    val_labels = dataset_dir / "labels" / "val"

    for folder in (train_images, train_labels, val_images, val_labels):
        folder.mkdir(parents=True, exist_ok=True)

    source_dir = root / "ID"
    image_files = sorted(
        path
        for path in source_dir.glob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )

    if not image_files:
        raise FileNotFoundError(f"No image files were found in {source_dir}")

    if len(image_files) < 2:
        train_files = image_files
        val_files = image_files
    else:
        split_index = max(1, int(len(image_files) * 0.8))
        train_files = image_files[:split_index]
        val_files = image_files[split_index:]

    for image_path in train_files:
        shutil.copy2(image_path, train_images / image_path.name)
        (train_labels / f"{image_path.stem}.txt").write_text(
            "0 0.5 0.5 1.0 1.0\n",
            encoding="utf-8",
        )

    for image_path in val_files:
        shutil.copy2(image_path, val_images / image_path.name)
        (val_labels / f"{image_path.stem}.txt").write_text(
            "0 0.5 0.5 1.0 1.0\n",
            encoding="utf-8",
        )

    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_dir.as_posix()}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                "names: ['id']",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return data_yaml


def main():
    root = Path(__file__).resolve().parent
    data_path = ensure_dataset(root)

    # 1. Initialize a baseline model configuration
    # Loading 'yolo11n.pt' or 'yolov8n.pt' transfers pre-trained weights (Transfer Learning).
    # This speeds up training significantly compared to training from scratch ('yolo11n.yaml').
    model = YOLO(str(root / "yolo11n.pt"))

    # 2. Run the Training Pipeline
    print("Initiating training sequence...")
    print(f"Training dataset config: {data_path}")
    epochs = int(os.getenv("YOLO_EPOCHS", "50"))
    training_results = model.train(
        data=str(data_path),  # Path to your dataset configuration file
        epochs=epochs,  # Number of complete passes through the dataset
        imgsz=640,  # Input image size (resizes images to 640x640)
        batch=16,  # Number of images processed per hardware batch step
        device="cpu",  # Set to 'cuda' or 0 if running on an NVIDIA GPU
        workers=4,  # Number of CPU threads dedicated to loading data
        project="my_custom_yolo",  # The root output directory folder name
        name="run_v1",  # Subfolder storing weights and performance graphs
    )

    print("\n[SUCCESS] Training completed.")
    print(
        f"Best weights saved to: {os.path.join('my_custom_yolo', 'run_v1', 'weights', 'best.pt')}"
    )

    # 3. Model Evaluation & Validation
    # Automatically computes validation metrics against the validation set
    print("\nRunning evaluation metrics pass...")
    metrics = model.val()

    # Print core Mean Average Precision (mAP) validation scores
    print(f"mAP50 Accuracy Score   : {metrics.box.map50:.4f}")
    print(f"mAP50-95 Accuracy Score: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()