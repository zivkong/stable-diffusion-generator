# Stable Diffusion Generator Project

Stable Diffusion Generator is a project designed to generate and manipulate images using advanced AI models, such as Stable Diffusion. The project includes a generator module for creating images and a web-based user interface for interacting with the system.

Stable Diffusion Generator/

## Project Structure
```
./
├── requirements.txt       # Python dependencies
├── data/                  # Data storage (e.g., logs, input prompts)
│   └── prompt_log.json    # Log of prompts used for image generation
├── env/                   # Python virtual environment
├── output/                # Generated images
├── src/                   # Source code
│   ├── generator/         # Image generation logic
│   │   ├── __init__.py
│   │   └── sd_generator.py
│   ├── ui/                # Web-based user interface
│   │   ├── __init__.py
│   │   └── web_ui.py
```

## Key Components

### 1. Generator Module
- **File**: `src/generator/sd_generator.py`
- **Purpose**: Handles image generation using Stable Diffusion models.
- **Key Classes**:
  - `StableDiffusionGenerator`: Initializes and manages the Stable Diffusion pipeline.

### 2. Web UI Module
- **File**: `src/ui/web_ui.py`
- **Purpose**: Provides a web-based interface for users to interact with the generator.
- **Key Functions**:
  - `get_device()`: Determines the best available device (e.g., GPU) for computation.

## Development Guidelines

### Setting Up the Environment
1. Create a virtual environment:
   ```bash
   python3 -m venv env
   ```
2. Activate the virtual environment:
   ```bash
   source env/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root and specify the address and port:
   ```env
   UI_ADDRESS=127.0.0.1
   UI_ADDRESS=8000
   ```

### Running the Project
1. Start the web UI with Uvicorn (recommended):
   ```bash
   uvicorn src.ui.uvicorn_config:app --reload
   ```
   Or, for production:
   ```bash
   uvicorn src.ui.uvicorn_config:app --host 0.0.0.0 --port 8000
   ```
2. (Alternative) Start the web UI directly (for debugging):
   ```bash
   python src/ui/web_ui.py
   ```
3. Access the interface in your browser at the address and port specified in the `.env` file (default: http://127.0.0.1:8000).

### Logging
- Logs are written to `output/logs.log` for debugging and monitoring purposes.

### Future Development
- **Logging**: Enhance logging to include more detailed metrics and error tracking.
- **Testing**: Add unit tests for both the generator and UI modules.
- **Features**: Implement additional image manipulation features, such as inpainting and style transfer.

## Always Remember
- Always update this README file with any changes to the project structure, features, or development guidelines.

## Notes for AI Assistants
For detailed notes and guidelines tailored for AI systems, please refer to the [GUIDELINES.md](./GUIDELINES.md) file.

### Apple Silicon (M1/M2/M3) Acceleration
If you are using a Mac with Apple Silicon, you can enable GPU acceleration using the Metal Performance Shaders (MPS) backend for PyTorch. This will significantly speed up image generation compared to CPU-only mode.

#### Steps for Apple Silicon Users
1. (Recommended) Create and activate a new virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```
2. Install the Apple Silicon compatible requirements:
   ```bash
   pip install -r requirements_mps.txt
   ```
   This will install the correct versions of PyTorch and diffusers for MPS support.
3. Run the project as usual. The generator will automatically use MPS if available.

> **Note:**
> - You must be using Python 3.8 or newer.
> - If you see a warning about running on CPU, check your PyTorch installation and that your Mac supports MPS.
> - For best results, keep your macOS and Python packages up to date.
