# Project Aether's Eye 👁️✨

![Project Status](https://img.shields.io/badge/status-in%20development-blue)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)
![PyTorch](https://img.shields.io/badge/PyTorch-1.x-EE4C2C?logo=pytorch)
![License](https://img.shields.io/badge/license-MIT-green)



**Project Aether's Eye is an autonomous AI system designed to monitor astronomical time-lapse data, intelligently detect anomalies, and prioritize discoveries for further analysis.**

---

## 📖 Table of Contents

* [About The Project](#about-the-project)
* [🚀 Key Features](#-key-features)
* [🛠️ Technology Stack](#️-technology-stack)
* [🔬 System Architecture](#-system-architecture)
* [🏁 Getting Started](#-getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
* [💻 Usage](#-usage)
* [🤝 Contributing](#-contributing)
* [📜 License](#-license)
* [📧 Contact](#-contact)

---

## About The Project

The universe is filled with transient events—supernovae, passing satellites, meteors, and phenomena yet to be discovered. Sifting through terabytes of astronomical footage is a monumental task for humans. Aether's Eye was built to automate this process.

This project implements a multi-stage AI pipeline that not only detects changes in the night sky but also understands their significance. It uses a **dual-AI core**, leveraging both TensorFlow and PyTorch, to first classify potential anomalies and then score them based on characteristics like speed, brightness, and trajectory, simulating how a human astronomer might prioritize a new discovery.

---

## 🚀 Key Features

* **🤖 Autonomous Detection Pipeline:** Ingests raw time-lapse video and outputs a prioritized list of astronomical events.
* **🧠 Dual-AI Core:** Uses a **TensorFlow-based CNN** for noise filtering and a **PyTorch-based Neural Network** for intelligent event prioritization.
* **⭐ Smart Prioritization:** Goes beyond simple detection to score anomalies, allowing users to focus on the most scientifically interesting events.
* **🖥️ Real-Time Simulation:** Processes frames sequentially to simulate a live data feed from a telescope.
* **📊 Visualization Ready:** The final stage is designed to feed data into a "Mission Control" dashboard for real-time monitoring.

---

## 🛠️ Technology Stack

This project is built with a powerful stack of open-source data science and computer vision libraries:

* **Core Language:** [Python](https://www.python.org/)
* **AI & Machine Learning:**
    * [TensorFlow](https://www.tensorflow.org/): For building the CNN that classifies anomalies vs. sensor noise.
    * [PyTorch](https://pytorch.org/): For the regression model that assigns priority scores to events.
* **Computer Vision:** [OpenCV](https://opencv.org/): For all image processing tasks, from video-to-frame conversion to feature extraction.
* **Data Handling:**
    * [Pandas](https://pandas.pydata.org/): For organizing and logging extracted event data.
    * [NumPy](https://numpy.org/): For efficient numerical operations.
* **Data Visualization:** [Matplotlib](https://matplotlib.org/): For the final dashboard and data analysis.

---

## 🔬 System Architecture

The project operates on a five-phase pipeline, where the output of each phase serves as the input for the next. This modular design ensures a clean and logical data flow.



1.  **⚙️ Phase 1: Setup and Data Preparation**
    * The environment is configured, and the source time-lapse video is ingested. An OpenCV script then breaks the video down into thousands of sequential image frames.

2.  **🤖 Phase 2: The Sentinel (Detection & Classification)**
    * Consecutive frames are compared to detect pixel-level changes. A pre-trained **TensorFlow CNN** then classifies these changes, filtering out sensor noise and identifying true potential anomalies.

3.  **✨ Phase 3: The Cognitive Core (Prioritization)**
    * For each verified anomaly, key features (brightness, size, velocity, trajectory) are extracted. This feature vector is fed into a **PyTorch Neural Network** which outputs a priority score from 0.0 to 1.0.

4.  **🛰️ Phase 4: Integration & Simulation**
    * All components are chained together in a master script. This loop processes frames sequentially, simulating a real-time data stream and triggering alerts for high-priority events.

5.  **📊 Phase 5: Visualization Dashboard**
    * The processed data and high-priority alerts are logged and prepared for visualization. This phase feeds a Matplotlib or Plotly-based dashboard, acting as the mission control interface for monitoring the AI's findings.

---

## 🏁 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Ensure you have the following installed on your system:
* Git
* Python 3.9 or higher
* `pip` and `venv`

### Installation

1.  **Clone the repository**
    ```sh
    git clone [https://github.com/your-username/project-aethers-eye.git](https://github.com/your-username/project-aethers-eye.git)
    cd project-aethers-eye
    ```

2.  **Create and activate a virtual environment**
    * On macOS & Linux:
        ```sh
        python3 -m venv venv
        source venv/bin/activate
        ```
    * On Windows:
        ```sh
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install the required packages**
    ```sh
    pip install -r requirements.txt
    ```

---

## 💻 Usage

Once the installation is complete, you can run the main processing pipeline.

1.  Place your astronomical time-lapse video file (e.g., `night_sky.mp4`) into a `data/video` directory.
2.  Run the main script from the root of the project directory:
    ```sh
    python main.py --video data/video/night_sky.mp4
    ```
3.  The system will begin processing the video frame by frame. Watch the console for real-time updates and high-priority alerts. The final output, including a CSV log of all detected events, will be saved in the `output` directory.

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📧 Contact

Jamie Abrahams - [GitHub Profile](https://github.com/Jamie123234)

Project Link: [https://github.com/your-username/project-aethers-eye](https://github.com/Jamie123234/project_aether)
