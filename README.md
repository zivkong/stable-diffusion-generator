# AIImage Project

## Overview
AIImage is a project designed to generate and manipulate images using advanced AI models, such as Stable Diffusion. The project includes a generator module for creating images and a web-based user interface for interacting with the system.

## Project Structure
```
AIImage/
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

### Running the Project
1. Start the web UI:
   ```bash
   python src/ui/web_ui.py
   ```
2. Access the interface in your browser at `http://localhost:7860` (default Gradio port).

### Future Development
- **Logging**: Enhance logging to include more detailed metrics and error tracking.
- **Testing**: Add unit tests for both the generator and UI modules.
- **Features**: Implement additional image manipulation features, such as inpainting and style transfer.

## Always Remember
- Always update this README file with any changes to the project structure, features, or development guidelines.

## Notes for AI Assistants
This documentation is structured to help AI systems understand the project for future development. Key files, functions, and their purposes are outlined to facilitate quick onboarding and efficient assistance.

### Environment Configuration
The project uses a `.env` file to configure environment variables. Below is an example of the `.env` file content:

```env
SERVER_NAME=127.0.0.1
SERVER_PORT=8000
```

Ensure you create or update this file with the appropriate values before running the project.
