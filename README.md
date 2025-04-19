# Stable Diffusion Generator Project

Stable Diffusion Generator is a project designed to generate and manipulate images using advanced AI models, such as Stable Diffusion. The project includes a generator module for creating images and a web-based user interface for interacting with the system.

---

## Project Structure
```
./
├── requirements.txt       # Python dependencies
├── requirements_mps.txt   # Dependencies for MPS (Metal Performance Shaders) on macOS
├── data/                  # Data storage (e.g., logs, input prompts)
│   ├── chat_history.json  # Chat history for user interactions
│   └── prompt_log.json    # Log of prompts used for image generation
├── env/                   # Python virtual environment
├── models/                # Pre-trained models and related files
├── output/                # Generated images and logs
├── src/                   # Source code
│   ├── generator/         # Image generation logic
│   │   ├── config.py      # Configuration for the generator
│   │   └── sd_generator.py
│   ├── ui/                # Web-based user interface
│   │   ├── uvicorn_config.py # Uvicorn configuration for FastAPI
│   │   ├── web_ui.py      # Gradio-based web interface
│   │   └── static/        # Static files (e.g., fonts, images)
├── tests/                 # Unit tests
```

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd AIImage
```

### 2. Set Up the Virtual Environment
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 3. Run the Application
Ensure the Python virtual environment is set up and the required dependencies are installed. Then run:
```bash
uvicorn src.ui.app:app --reload
```
This will start the FastAPI server for the web-based user interface. You can access it in your browser at `http://127.0.0.1:8000`.

---

## Key Components

### Generator Module
- **Purpose**: Handles the logic for generating images using Stable Diffusion.
- **Files**:
  - `config.py`: Contains configuration settings for the generator.
  - `sd_generator.py`: Implements the main image generation logic.

### Web UI
- **Purpose**: Provides a Gradio-based interface for interacting with the generator.
- **Integration**: Built with FastAPI for additional API capabilities and hot reloading.
- **Files**:
  - `web_ui.py`: Main script for the Gradio-based web interface.
  - `uvicorn_config.py`: Configuration for running the FastAPI server.

### Electron Integration
- **Purpose**: Offers a desktop application interface for the Stable Diffusion Generator.
- **Features**:
  - Launches an Electron window to load the web-based UI served by the Python server.
  - Automatically starts the Python server when the Electron app is launched.
  - Includes debugging tools such as logging and DevTools.
- **Workflow**:
  1. Activates the Python virtual environment.
  2. Starts the Python server (`web_ui.py`).
  3. Loads the UI in the Electron window from `http://127.0.0.1:8000`.
  4. Terminates the Python server when the app is closed.

---

## Notes

- For macOS users with MPS support, use `requirements_mps.txt` to install dependencies optimized for Metal Performance Shaders.
- Logs and generated images are stored in the `output/` directory.
