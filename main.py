import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThread, pyqtSignal
from gui import MainWindow
from huggingface_api import HuggingFaceAPI

class WorkerThread(QThread):
    result_signal = pyqtSignal(list)
    progress_signal = pyqtSignal(int)
    message_signal = pyqtSignal(str, str)
    inference_result_signal = pyqtSignal(dict)

    def __init__(self, api):
        super().__init__()
        self.api = api
        self.task = None
        self.args = None

    def run(self):
        if self.task == "search":
            results = self.api.search_models(*self.args)
            self.result_signal.emit(results)
        elif self.task == "download":
            try:
                def progress_callback(progress):
                    self.progress_signal.emit(progress)
                
                filepath = self.api.download_model(*self.args, progress_callback=progress_callback)
                self.message_signal.emit("Download Complete", f"Model downloaded to: {filepath}")
            except Exception as e:
                self.message_signal.emit("Download Error", str(e))
                self.progress_signal.emit(0)
        elif self.task == "inference":
            try:
                result = self.api.run_inference(*self.args)
                self.inference_result_signal.emit(result)
            except Exception as e:
                self.message_signal.emit("Inference Error", str(e))

    def search(self, query, filters):
        self.task = "search"
        self.args = (query, filters)
        self.start()

    def download(self, model_id, download_dir):
        self.task = "download"
        self.args = (model_id, download_dir)
        self.start()

    def inference(self, model_id, inputs):
        self.task = "inference"
        self.args = (model_id, inputs)
        self.start()

class Settings:
    def __init__(self):
        self.api_key = ""
        self.default_download_dir = ""
        self.theme = "light"
        self.load_settings()

    def load_settings(self):
        if os.path.exists("settings.txt"):
            with open("settings.txt", "r") as f:
                lines = f.readlines()
                if len(lines) >= 3:
                    self.api_key = lines[0].strip()
                    self.default_download_dir = lines[1].strip()
                    self.theme = lines[2].strip()

    def save_settings(self):
        with open("settings.txt", "w") as f:
            f.write(f"{self.api_key}\n{self.default_download_dir}\n{self.theme}")

def main():
    app = QApplication(sys.argv)
    
    settings = Settings()
    api = HuggingFaceAPI(settings.api_key)
    
    window = MainWindow()
    worker = WorkerThread(api)

    # Connect signals
    window.theme_signal.connect(lambda theme: setattr(settings, 'theme', theme))
    window.search_signal.connect(worker.search)
    window.download_signal.connect(worker.download)
    window.inference_signal.connect(worker.inference)
    worker.result_signal.connect(window.update_results)
    worker.progress_signal.connect(window.update_progress)
    worker.message_signal.connect(window.show_message)
    worker.inference_result_signal.connect(window.update_inference_output)

    # Connect settings signals
    window.api_key_signal.connect(lambda key: setattr(settings, 'api_key', key))
    window.api_key_signal.connect(lambda key: setattr(api, 'api_key', key))
    window.default_dir_signal.connect(lambda dir: setattr(settings, 'default_download_dir', dir))

    # Load initial settings
    window.api_key_input.setText(settings.api_key)
    window.download_dir_input.setText(settings.default_download_dir)
    window.theme_selector.setCurrentIndex(1 if settings.theme == "dark" else 0)

    # Save settings on close
    app.aboutToQuit.connect(settings.save_settings)

    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
