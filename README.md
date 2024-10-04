# Hugging Face Hub App

## Overview

Hugging Face Hub App is a desktop application built with PyQt6 that allows users to search, download, and run inferences on models from the Hugging Face Model Hub. The application provides a user-friendly graphical interface to interact with the Hugging Face API.

## Features

- **Search Models**: Search for models on the Hugging Face Model Hub using various filters.
- **Download Models**: Download selected models to a specified directory.
- **Run Inference**: Perform inference using downloaded models directly from the application.
- **Settings Management**: Configure API keys and default download directories.

## Installation

### Prerequisites

- Python 3.7 or higher
- A Hugging Face API key

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/mixelpixx/HF-APP
   cd mixelpixx/HF-APP
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python main.py
   ```

## Usage

1. **Launch the Application**: Start the application by running `python main.py`.
2. **Search for Models**: Use the Search tab to find models by entering a query and applying filters.
3. **Download Models**: Select a model from the search results and click 'Download Selected'.
4. **Run Inference**: Navigate to the Inference Playground tab, enter a model ID and input data, then click 'Run Inference'.
5. **Configure Settings**: Go to the Settings tab to set your API key and default download directory.

## Configuration

- **API Key**: Enter your Hugging Face API key in the Settings tab and save it.
- **Download Directory**: Specify a default directory for model downloads.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push to your fork.
4. Submit a pull request with a description of your changes.

## License

This project is licensed under the MIT License.

## Contact

For questions or support, please contact [your-email@example.com].

## Acknowledgments

- [PyQt6](https://riverbankcomputing.com/software/pyqt/intro) for the GUI framework.
- [Hugging Face](https://huggingface.co/) for the API and model hub.
