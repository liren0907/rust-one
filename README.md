# Rust One Toolbox

A Rust learning project featuring data science, machine learning, and computer vision tutorial modules.

## Project Structure

```
rust-one/
├── src/                    # Main program entry
├── crates/
│   ├── csv-converter/      # CSV conversion utilities
│   ├── data-science/       # Data science & statistical analysis
│   ├── opencv-tutorial/    # OpenCV computer vision tutorials
│   ├── polars-tutorial/    # Polars data processing tutorials
│   └── ort-tutorial/       # ONNX Runtime inference (YOLO)
├── data/                   # Test data
├── model_weight/           # Model weight files
└── output/                 # Output results
```

## Requirements

- Rust 2024 Edition
- OpenCV (for opencv-tutorial and ort-tutorial)
- ONNX Runtime (for ort-tutorial)


## Quick Start

```bash
# Build the entire workspace
cargo build

# Run the main program
cargo run

# Run data-science examples
cargo run -p data-science --example data_science_demo
cargo run -p data-science --example statistics_tutorial
cargo run -p data-science --example machine_learning_demo

# Run ort-tutorial (YOLO object detection)
cargo run -p ort-tutorial --release
```

## Crates Overview

| Crate | Description |
|-------|-------------|
| **csv-converter** | CSV file processing and conversion utilities (CSV → JSON) |
| **data-science** | Statistics, probability distributions, hypothesis testing, ML algorithms (KNN, Decision Tree) |
| **opencv-tutorial** | OpenCV computer vision tutorials and image processing |
| **polars-tutorial** | High-performance data processing using Polars |
| **ort-tutorial** | ONNX Runtime inference with YOLO object detection |
