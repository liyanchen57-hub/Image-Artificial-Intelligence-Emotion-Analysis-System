# Realtime Facial Emotion Recognition

This project is a complete end-to-end facial emotion recognition system built on FER2013.

It includes:

- offline model training
- evaluation and visualization
- realtime webcam inference  
- web-based realtime dashboard

## Features

- Train 7-class facial emotion recognition models on FER2013
- Support both `cnn_cbam` and `resnet_cbam`
- Export loss curves, accuracy curves, confusion matrix, and classification report
- Support `cpu`, `cuda`, and `mps`
- Run realtime webcam emotion recognition

- Run a web-based realtime dashboard with:
  - browser-based video stream
  - face detection and emotion label
  - emotion probability bars
  - session summary
  - realtime emotion curves

## Emotion Classes

The project uses the 7 FER2013 emotion classes:

- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

## Project Structure

```text
.
├── data/
│   ├── train/
│   └── test/
├── outputs/
├── src/
│   ├── data/
│   │   └── fer2013_dataset.py
│   ├── engine/
│   │   └── trainer.py
│   ├── inference/
│   │   ├── realtime.py
│   │   └── web_app.py
│   ├── models/
│   │   ├── attention.py
│   │   ├── cnn_cbam.py
│   │   ├── factory.py
│   │   └── resnet_cbam.py
│   └── utils/
│       ├── metrics.py
│       └── visualization.py
├── train.py
├── plot_class_distribution.py
├── web_demo.py
├── requirements.txt
└── README.md
```

## Environment

Recommended environment:

- Python 3.10 or later
- Windows or Linux with NVIDIA CUDA
- CUDA-enabled PyTorch installation recommended
- CPU is also supported but training will be slower

Install dependencies:

```bash
pip install -r requirements.txt
```

To check CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Dataset

The dataset is expected in directory form:

```text
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

The training split is further divided into training and validation sets automatically.

## Dataset Visualization

To show the class imbalance problem in FER2013, run:

```bash
python plot_class_distribution.py
```

## Training

### Train the CNN-CBAM model with Focal Loss

```bash
python train.py --data-root data --output-dir outputs/full_run_focal_gamma1_cnn_cbam_cuda --model cnn_cbam --epochs 25 --batch-size 64 --num-workers 0 --device cuda
```

### Train the improved ResNet-CBAM model with Focal Loss

The current recommended model is `resnet_cbam`.

```bash
python train.py --data-root data --output-dir outputs/full_run_focal_gamma1_resnet_cbam_cuda --model resnet_cbam --epochs 25 --batch-size 64 --num-workers 0 --device cuda
```

### Main arguments

- `--model`: model architecture (`cnn_cbam` or `resnet_cbam`)
- `--device`: computation device (`auto`, `cpu`, `cuda`, or `mps`)
- `--epochs`: number of training epochs
- `--batch-size`: batch size
- `--early-stopping-patience`: early stopping patience

The current training script uses Focal Loss with:

- `gamma = 1.0`
- `alpha = None`

## Training Outputs

Each training run exports:

- `checkpoints/best_model.pth`
- `reports/history.csv`
- `reports/classification_report.txt`
- `reports/summary.json`
- `plots/training_curves.png`
- `plots/confusion_matrix.png`

## Current Best Result

Among the tested models, `resnet_cbam` achieved the best overall performance.

Current best result with `resnet_cbam`:

- device: `cuda`
- epochs: `25`
- best epoch: `24`
- best validation accuracy: `0.6660`
- final test accuracy: `0.6567`


Test-set summary:

              precision    recall  f1-score   support

       angry     0.6220    0.5428    0.5797       958
     disgust     0.6042    0.5225    0.5604       111
        fear     0.5524    0.3760    0.4474      1024
       happy     0.8514    0.8788    0.8649      1774
     neutral     0.5721    0.6594    0.6127      1233
         sad     0.4976    0.5734    0.5328      1247
    surprise     0.7721    0.7990    0.7853       831

    accuracy                         0.6567      7178
   macro avg     0.6388    0.6217    0.6262      7178
weighted avg     0.6557    0.6567    0.6523      7178


### Web dashboard

Run the web version:

```bash
python web_demo.py --checkpoint outputs/full_run_focal_gamma1_resnet_cbam_cuda/checkpoints/best_model.pth --model resnet_cbam --device cuda
```

Then open:

```text
http://127.0.0.1:7860
```

Optional web arguments:

- `--host`: server host, default `127.0.0.1`
- `--port`: server port, default `7860`
- `--camera`: camera index, default `0`
- `--frame-width`: camera frame width
- `--frame-height`: camera frame height
- `--jpeg-quality`: MJPEG stream quality
- `--history-length`: number of emotion samples kept for the curve

## Realtime Interface

The web-based dashboard includes:

- browser-based webcam stream
- real-time face detection with bounding boxes
- main emotion prediction label
- emotion probability bars
- session summary statistics
- real-time emotion trend curves

The interface is designed with a clean dark theme for better visualization.

## Face Detection

The realtime pipeline currently uses a layered strategy:

- use old `MediaPipe solutions` API when available
- otherwise fall back to `OpenCV Haar Cascade`

This is used to keep the project compatible across different `mediapipe` versions.

## Engineering Notes

- Training and realtime inference are separated
- Models are created through a unified factory
- The training pipeline includes progress bars, learning-rate scheduling, early stopping, and exported visualizations

## Future Work

Possible next improvements:

- replace the fallback detector with RetinaFace or YOLO-face
- use a true time-based x-axis for realtime curves
- export realtime session logs as CSV or JSON
- support multi-face tracking
- add richer web interaction controls
