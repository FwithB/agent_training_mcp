# YOLO & CVAT Automation System - Iteration 5

[English](README.md) | [中文](README_CN.md)

A comprehensive machine learning workflow management system based on MCP architecture, integrating YOLO model training, CVAT data annotation, model deployment, and automated inference capabilities.

## Overview

Iteration 5 builds upon Iteration 4 to deliver a complete end-to-end machine learning workflow automation system. The system now supports the entire ML pipeline from data upload, annotation, training, deployment to inference. Beyond YOLO model training and browser automation, it features complete CVAT API integration, model deployment services, and data format conversion capabilities.

## System Architecture

```
Client (main.py) <---> Server (server.py)
                           |
                           +---> Training Module (train.py)
                           +---> Browser Module (browser.py)
                           +---> CVAT API Module (cvat_api.py)
                           +---> Deployment Module (deploy_to_cvat.py)
                           |
                           v
                    Nuclio Services <--> CVAT System
```

## Core Components

### Main Modules
- **main.py**: Client application with 11 functional options
- **server.py**: MCP server handling client requests and module coordination
- **train.py**: YOLO model training module
- **browser.py**: Browser automation control module
- **cvat_api.py**: Complete CVAT system API wrapper
- **deploy_to_cvat.py**: Model deployment to Nuclio services
- **config.py**: System configuration management

### Support Modules
- **test_model.py**: Model testing and validation
- **deploy_templates/**: Nuclio deployment templates
  - **function.yaml**: Nuclio function configuration template
  - **main.py**: Inference service code template

### Data Directories
- **datasets/**: Converted YOLO format training data
- **downloads/**: Raw annotation data downloaded from CVAT
- **runs/**: Model training outputs and weight files
- **coco128/**: Sample dataset

## Key Features vs Iteration 4

| Feature | Iteration 4 | Iteration 5 |
|---------|-------------|-------------|
| Functionality | YOLO training + Browser automation | Complete ML workflow platform |
| Client Options | 4 basic functions | 11 comprehensive modules |
| Core Components | 4 (main.py, server.py, train.py, browser.py) | 8 (added cvat_api.py, deploy_to_cvat.py, config.py, test_model.py) |
| Data Management | No data management capabilities | Complete data lifecycle management |
| Model Deployment | No deployment features | Nuclio microservice auto-deployment |
| CVAT Integration | Basic browser control | Complete API integration and automation |
| Workflow Integration | Partial workflow | End-to-end closed-loop workflow |
| Natural Language | Basic command parsing | Complex workflow instruction support |

## Client Functionality Overview

The system provides 11 main functional options:

### 1. Natural Language Instruction Control
Execute single operations through natural language dialogue:
- **Training commands**: "Train yolov8n model using latest dataset for 5 epochs"
- **Deployment commands**: "Deploy latest trained model to CVAT"
- **Data commands**: "Upload data to CVAT and create new task"
- **Annotation commands**: "Run auto annotation on task 14"
- **Download commands**: "Download dataset from task 12"
- **Conversion commands**: "Convert recently downloaded dataset to YOLO format"

### 2. Train YOLO Model (Manual Parameters)
Precise control over model training process:
- Select model type (yolov8n/s/m/l/x)
- Set training epochs
- Specify dataset configuration file
- Auto-discover latest converted datasets
- Real-time training progress and results

### 3. Browser Control (Manual Parameters)
Browser automation operations:
- Open browser and navigate to CVAT
- Auto-login to CVAT system
- Navigate to specific URLs
- Create CVAT projects
- Force browser restart
- Support Chrome, Edge, Firefox

### 4. Deploy Model to CVAT
Deploy trained models as inference services:
- Auto-discover latest trained model weights
- Generate Nuclio function configurations
- Configure GPU resources and dependencies
- Deploy as scalable microservices
- Seamless CVAT system integration

### 5. Upload Data to CVAT
Create annotation tasks and upload data:
- Auto-create CVAT tasks
- Upload image data to tasks
- Configure COCO 80-class label system
- Support batch image uploads
- Automatic compression and format conversion

### 6. Model Inference/Auto Annotation
Use deployed models for automatic annotation:
- Select deployed inference models
- Specify CVAT tasks for annotation
- Configure confidence thresholds
- Set batch processing sizes
- Monitor annotation progress and results

### 7. View Annotation Interface
Open CVAT annotation interface:
- Specify task ID and job ID
- Auto-login to CVAT system
- Direct navigation to annotation interface
- Support multi-task switching
- Browser state management

### 8. Download Dataset
Export annotation data from CVAT:
- Select export formats (CVAT, COCO, Datumaro, etc.)
- Specify output paths
- Choose whether to include image files
- Automatic packaging and download
- Support large dataset batch processing

### 9. Convert Dataset Format
Convert CVAT data to YOLO format:
- Auto-identify recently downloaded datasets
- Parse multiple annotation types (boxes, polygons, rotated boxes)
- Intelligent dataset splitting (train/validation)
- Generate YOLO configuration files
- Data validation and cleaning

### 10. One-Click Download and Convert Dataset
Automated data preparation workflow:
- Specify CVAT task ID
- Auto-download annotation data
- Immediately convert to YOLO format
- Generate ready-to-use training configurations
- Complete operation logging

### 11. Exit Program
Safely exit the client application

## Technical Implementation

### MCP Architecture Extensions
- Added `/deploy` endpoint for model deployment requests
- Added `/open_annotation` endpoint for opening annotation interface
- Extended JSON communication protocol for complex parameter passing
- Improved error handling and status reporting mechanisms

### CVAT API Integration
- Complete CVAT REST API wrapper
- Support for task creation, data upload, model management
- Automated annotation requests and status monitoring
- Data download and format conversion capabilities

### Model Deployment Automation
- Template-based Nuclio function generation
- Automatic Docker image and dependency configuration
- GPU resource management and performance optimization
- Service health checks and fault recovery

### Data Format Conversion
- Support for bounding boxes, polygons, rotated boxes, and other annotation types
- Intelligent coordinate system conversion and normalization
- Automatic dataset validation and cleaning
- Standard YOLO configuration file generation

## Requirements

### Python Dependencies
```
ultralytics>=8.0.0    # YOLO model training
flask>=2.0.0          # HTTP server
requests>=2.25.0      # HTTP client
openai>=1.0.0         # LLM API calls
playwright>=1.30.0    # Browser automation
tqdm>=4.62.0          # Progress bars
python-dotenv>=0.19.0 # Environment variable management
```

### System Dependencies
- Python 3.8+
- Docker (for CVAT and Nuclio)
- WSL (Linux compatibility layer for Windows)

### External Services
- CVAT annotation system (deployed via Docker)
- Nuclio serverless computing platform
- OpenRouter API (for natural language processing)

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd yolo-cvat-automation
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
python -m playwright install
```

### 3. Setup CVAT Environment
```bash
# Download and start CVAT
git clone https://github.com/opencv/cvat.git
cd cvat
docker-compose up -d
```

### 4. Configure API Keys
Set API key in `main.py` or via environment variable:
```bash
export OPENROUTER_API_KEY=your_api_key_here
```

### 5. Configure CVAT Connection
Edit `config.py` to set CVAT connection parameters:
```python
CVAT_URL = "http://localhost:8080"
CVAT_USERNAME = "admin"
CVAT_PASSWORD = "your_password"
```

## Quick Start

### 1. Start Server
```bash
python server.py
```

### 2. Run Client
```bash
python main.py
```

### 3. Choose Operation Mode
- For new users, natural language instruction mode is recommended
- For precise control, use manual parameter mode

## Workflow Examples

### Scenario 1: Custom Model Training
```
1. Upload data to CVAT → Select menu option 5
2. Manually annotate in CVAT
3. Download and convert dataset → Select menu option 10
4. Train YOLO model → Select menu option 2
5. Deploy model to CVAT → Select menu option 4
```

### Scenario 2: Pre-trained Model Annotation
```
1. Upload data for annotation → Select menu option 5
2. Run auto annotation → Select menu option 6
3. Review annotation results → Select menu option 7
4. Download annotated data → Select menu option 8
```

### Scenario 3: Natural Language Control
```
Input: "Download dataset from task 14"
System executes: Automatically downloads annotation data from specified CVAT task

Input: "Convert recently downloaded dataset"
System executes: Converts downloaded CVAT data to YOLO training format

Input: "Train yolov8n model using latest dataset"
System executes: Uses converted dataset to train specified model
```

## Advanced Features

### Custom Model Deployment
The system automatically discovers latest trained model weights and generates corresponding Nuclio services:
- Automatic GPU resource configuration
- Unique service name generation
- Integrated COCO 80-class label system

### Dataset Management
- Auto-identify and recommend latest converted datasets
- Support intelligent conversion of multiple data formats
- Dataset version management and history tracking

### Batch Operations
- Support multi-task parallel processing
- Batch data download and conversion
- Automatic retry and error recovery

## Troubleshooting

### Common Issues

**1. CVAT Connection Failed**
- Check if Docker service is running properly
- Verify CVAT container status: `docker ps`
- Ensure port 8080 is not occupied

**2. Browser Automation Failed**
- Try different browser types
- Check if Playwright browser drivers are fully installed
- Ensure system has graphical interface environment

**3. Model Deployment Failed**
- Check if Nuclio service is running properly
- Verify model file paths and permissions
- Check deployment logs for detailed error information

**4. Natural Language Parsing Errors**
- Verify OpenRouter API key configuration
- Check network connection status
- Use manual parameter mode as fallback

### Log Files
The system generates detailed log files:
- `mcp_client.log` - Client operation logs
- `mcp_server.log` - Server runtime logs
- `browser_controller.log` - Browser operation logs

### Debug Mode
In development environment, enable debug mode for more detailed information:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Performance Optimization

### Hardware Recommendations
- **CPU**: 8+ cores for multi-task parallel processing
- **Memory**: 16GB+ for large dataset processing
- **GPU**: NVIDIA graphics card with CUDA acceleration for training and inference
- **Storage**: SSD for improved data I/O performance

### System Tuning
- Adjust batch sizes to optimize GPU utilization
- Configure appropriate worker process counts
- Use data caching to reduce redundant I/O operations

## Extension Development

### Adding New Data Formats
Extend the `FORMAT_MAP` dictionary in `cvat_api.py`:
```python
FORMAT_MAP["CUSTOM_FORMAT"] = "Custom Format 1.0"
```

### Integrating New Model Architectures
Modify `train.py` and deployment templates to support other deep learning frameworks.

### Custom Annotation Types
Extend conversion functionality to support point annotations, line annotations, and other annotation types.

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome through:
- Submitting issues to report problems
- Submitting pull requests to improve code
- Improving documentation and usage examples
- Sharing usage experiences and best practices

## Acknowledgments

- Built with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- Integrated with [CVAT](https://github.com/opencv/cvat)
- Powered by [Playwright](https://playwright.dev/) for browser automation
- Uses [Nuclio](https://nuclio.io/) for serverless model deployment