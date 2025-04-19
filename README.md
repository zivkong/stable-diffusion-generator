# Stable Diffusion Generator Project

Stable Diffusion Generator is a project designed to generate and manipulate images using advanced AI models, such as Stable Diffusion. The project includes a generator module for creating images and a web-based user interface for interacting with the system.

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
│   │   ├── __init__.py
│   │   ├── config.py      # Configuration for the generator
│   │   └── sd_generator.py
│   ├── ui/                # Web-based user interface
│   │   ├── __init__.py
│   │   ├── uvicorn_config.py # Uvicorn configuration for FastAPI
│   │   ├── web_ui.py      # Gradio-based web interface
│   │   └── static/        # Static files (e.g., fonts, images)
├── tests/                 # Unit tests
│   └── test_sd_generator.py
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd AIImage
   ```

2. **Set Up the Virtual Environment**
   ```bash
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the Application**
   Use the following command to start the Uvicorn server:
   ```bash
   PYTHONPATH=$(pwd) uvicorn src.ui.uvicorn_config:app --reload
   ```
   The application will be available at `http://127.0.0.1:8000`.

## Key Components

- **Generator Module**: Contains the logic for generating images using Stable Diffusion. Configurations are defined in `config.py`, and the main generation logic is in `sd_generator.py`.
- **Web UI**: A Gradio-based interface for interacting with the generator. It is integrated with FastAPI for hot reloading and additional API capabilities.

## Notes

- For macOS users with MPS support, use `requirements_mps.txt` to install dependencies optimized for Metal Performance Shaders.
- Logs and generated images are stored in the `output/` directory.
