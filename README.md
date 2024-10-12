# Arcade Flyers: AI-Driven Predictive Maintenance System

## Overview

Arcade Flyers is a robust AI-driven predictive maintenance system designed for aerospace fleets. By leveraging NVIDIA Inference Microservices (NIM), this solution processes real-time sensor data from aircraft systems to optimize maintenance scheduling and improve overall aircraft availability.

### Key Features

- **Dynamic Batching and Model Parallelism**: Optimize inference for large-scale predictive models.
- **Generative AI Insights**: Analyze historical and live data to generate actionable maintenance insights.
- **User-Friendly Interface**: Simplifies deployment for engineers, facilitating easy interaction with the system.
- **Real-Time Monitoring**: Ensures optimal performance and scalability of services.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Integration](#model-integration)
- [Features](#features)
- [Real-Time Monitoring](#real-time-monitoring)
- [Performance Metrics](#performance-metrics)
- [Future Work](#future-work)
- [License](#license)

## Installation

To run this application, you need to install the required Python packages. Use the following command:

```bash
pip install -r requirements.txt
```

Make sure to set up your environment variables, especially NVIDIA_API_KEY.

## Usage
Run the application using the following command:

```bash
streamlit run app.py
```

## Application Structure
# Damage Detection: 
Upload an image to check for dents or cracks using a YOLO model.
# Faulty Wire Detection: 
Enter voltage, current, and resistance values to predict wire faults.
# Fault Bot: 
Interact with a generative AI model to ask questions related to airplane accidents and maintenance.
## Model Integration
# YOLO Model
The application utilizes the YOLO model for detecting damages in aircraft images.
Preprocessing: Input images are resized and normalized for the model.
Postprocessing: The model's predictions are processed to extract bounding boxes and confidence scores.
## Generative AI Model
Provides responses to user queries based on historical data and context.
## Features
# Dynamic Batching: Processes multiple requests efficiently to optimize throughput.
# Model Parallelism: Utilizes multiple GPUs for handling larger models and higher workloads.
# Real-Time Monitoring
Includes real-time monitoring tools that allow users to track inference demand and auto-scale services based on performance metrics.

## Performance Metrics
# Inference Latency: Measures the time taken for the model to respond to requests.
# Throughput: Tracks the number of requests processed per second.
# Resource Utilization: Monitors CPU and GPU usage during inference.

## Future Work
Enhance the system to process real-time sensor data for a more comprehensive predictive maintenance solution.
Develop additional features for monitoring and analytics to improve user experience and system efficiency.
Implement detailed logging and error handling to improve system reliability.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
