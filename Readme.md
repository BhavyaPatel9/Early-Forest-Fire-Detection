# 🌲🔥 Early Forest Fire Detection System (IoT + UAV + AI)

![Project Banner](path/to/banner_image.png)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-ESP32%20%7C%20Raspberry%20Pi-green)](https://www.espressif.com/)
[![AI Model](https://img.shields.io/badge/AI-YOLOv8-orange)](https://github.com/ultralytics/ultralytics)
[![Status](https://img.shields.io/badge/Status-Completed-success)]()

## 📖 Overview
The **Early Forest Fire Detection System** is a hybrid safety architecture designed to minimize wildfire detection time and eliminate false alarms. It integrates a distributed **Wireless Sensor Network (WSN)** using ESP32 nodes with an autonomous **Unmanned Aerial Vehicle (UAV)** for visual verification.

### Key Features
* **Multi-Sensor IoT Nodes:** Real-time monitoring of Temperature, Humidity, Smoke (MQ-2), and Flame (IR).
* [cite_start]**Long-Range Communication:** Utilizes **ESP-NOW** protocol for efficient, grid-based sensor communication without reliance on cellular networks[cite: 53, 417].
* [cite_start]**Cloud Dashboard:** Centralized monitoring via Arduino IoT Cloud with threshold-based alerting logic[cite: 54, 446].
* **Autonomous Verification:** Automatically dispatches a UAV to alert coordinates upon threshold breach.
* [cite_start]**AI-Powered Detection:** Onboard Raspberry Pi runs a custom-trained **YOLOv8-n** model to visually confirm fire/smoke presence[cite: 117, 522].
* [cite_start]**Hexagonal Scanning:** Drone performs a multi-angle hexagonal scan of the target area to overcome canopy occlusion[cite: 501].

---

## 🏗️ System Architecture

The system operates in a **Two-Stage Verification Pipeline**:
1.  **Stage 1 (Sensing):** Ground nodes detect environmental anomalies and trigger a "Potential Fire" alert via the Cloud.
2.  **Stage 2 (Verification):** A drone flies to the specific GPS coordinates, scans the area, and uses Computer Vision to confirm the fire before notifying authorities.

![System Block Diagram](path/to/system_architecture_diagram.png)
---

## 📂 Project Structure

This repository is organized as follows:

```bash
EARLY_FOREST_FIRE_DETECTION_FYP/
│
├── 📂 Drone_mission_scripts/   # Python scripts for Autonomous Navigation
│   ├── drone_navigation.py     # Main script for takeoff, waypoints, and landing
│   └── hex_scan_logic.py       # Implementation of the Hexagonal Scanning Pattern
│
├── 📂 IOT_scripts/             # Firmware for ESP32 Nodes
│   ├── Sensor_Node_Code.ino    # Code for DHT11, MQ-2, Flame Sensor & ESP-NOW sender
│   └── Gateway_Node_Code.ino   # Code for ESP-NOW receiver & Cloud Uplink
│
├── 📂 Models/                  # Trained Deep Learning Models
│   ├── best.pt                 # The trained YOLOv8-n weights file
│   └── last.pt                 # Checkpoint weights
│
├── 📂 Models_scripts/          # Training & Inference Scripts
│   ├── train_yolo.ipynb        # Colab notebook used for training
│   └── real_time_inference.py  # Script to run on Raspberry Pi for live detection
│
└── 📄 Readme.md                # Project Documentation