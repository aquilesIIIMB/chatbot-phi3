# Phi-3 Mini On-Device AI Companion

A privacy-focused, personalized AI chatbot powered by Microsoft's Phi-3 mini language model that runs directly on your mobile device. This project implements a complete end-to-end solution from model optimization to app deployment.

Keywords: on-device AI, Phi-3 mini, model quantization, React Native, privacy-focused AI, edge ML deployment, personalized AI assistant

![Phi-3 Mini Chatbot](https://github.com/your-username/phi3-chatbot/raw/main/docs/images/app_screenshot.png)

## 🌟 Features

- **On-Device Inference**: Optimized Phi-3 mini model runs locally on your mobile device
- **Privacy-Focused**: All conversations stay on your device
- **Personalized Responses**: Adaptive conversation memory to provide personalized assistance
- **Cross-Platform**: Works on both iOS and Android
- **Cloud Infrastructure**: Flexible deployment options on GCP, AWS, or DataStax
- **Quantized Model**: Highly optimized for mobile performance with minimal quality loss
- **REST API Backend**: Scalable FastAPI implementation
- **Modern UI**: Clean, intuitive chat interface built with React Native

## 📋 Project Structure (WIP)

```
phi3_chatbot/
├── api/                # FastAPI backend
├── models/             # Model optimization scripts and model files
├── utils/              # Utility functions for conversations and model handling
├── config/             # Configuration files
├── tests/              # Test suite
├── mobile/             # React Native mobile application
│   ├── ios/            # iOS-specific code
│   └── android/        # Android-specific code
├── deployment/         # Deployment scripts for different clouds
│   ├── gcp/            # Google Cloud Platform deployment
│   ├── aws/            # AWS deployment
│   └── datastax/       # DataStax deployment
└── docs/               # Documentation
```

## 🚀 Getting Started (WIP)

### Prerequisites

- Python 3.10
- Node.js 14+
- React Native CLI
- Docker
- GCP (for cloud deployment)

### Model Preparation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/phi3-chatbot.git
   cd phi3-chatbot
   ```

2. Set up Python environment:
   ```bash
   python -m venv phi3_env
   source phi3_env/bin/activate  # On Windows: phi3_env\Scripts\activate
   pip install -r requirements.txt
   ```

3. Download and optimize the model:
   ```bash
   python models/download_and_optimize.py
   ```

### Backend Setup

1. Start the API server:
   ```bash
   cd api
   uvicorn main:app --reload
   ```

2. The API will be available at `http://localhost:8000`

### Mobile App Setup

1. Install dependencies:
   ```bash
   cd mobile
   npm install
   ```

2. Run on iOS:
   ```bash
   cd ios
   pod install
   cd ..
   npx react-native run-ios
   ```

3. Run on Android:
   ```bash
   npx react-native run-android
   ```

## 🛠️ Cloud Deployment (WIP)

### Google Cloud Platform (GCP)

1. Build and push Docker image:
   ```bash
   gcloud builds submit --tag us-central1-docker.pkg.dev/your-project/your-repo/phi3-api:v1
   ```

2. Deploy to Cloud Run:
   ```bash
   gcloud run deploy phi3-api --image us-central1-docker.pkg.dev/your-project/your-repo/phi3-api:v1
   ```

## 🧪 Testing

Run automated tests:
```bash
pytest tests/
```

## 📊 Performance Benchmarks (Expected - Simulated)

| Device | Inference Time | Memory Usage | Model Size |
|--------|---------------|--------------|------------|
| iPhone 13 | ~150ms | ~50MB | 485MB |
| Pixel 6 | ~180ms | ~60MB | 485MB |

## 🤔 How It Works

This project utilizes Microsoft's Phi-3 mini (4K context) model, optimized for on-device inference. The workflow includes:

1. **Model Quantization**: Reducing model size through 8-bit quantization
2. **Platform-Specific Optimization**: Converting to Core ML for iOS and TensorFlow Lite for Android
3. **Conversation Management**: Maintaining context for personalized responses
4. **API Layer**: Provides a consistent interface between the app and model
5. **Mobile Integration**: Native bridges to run the model efficiently on mobile devices

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please check out our [Contributing Guidelines](CONTRIBUTING.md).

## 📬 Contact

Questions? Reach out to [aquiles.martinez@ug.uchile.cl](mailto:aquiles.martinez@ug.uchile.cl)


## Relevant Commands
```bash
uv init chatbot-phi3
uv venv chatbot-phi3  
source chatbot-phi3/bin/activate

uv add --active torch transformers datasets evaluate accelerate jupyter matplotlib numpy pandas
uv add --active tensorboard onnx 'optimum[onnxruntime]' onnxruntime

uv add --active --dev ipykernel                
uv run --active ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=chatbot-phi3
uv add --active --dev uv 
```
