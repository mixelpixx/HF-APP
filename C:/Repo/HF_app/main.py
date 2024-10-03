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
        try:
            if self.task == "search":
                results = self.api.search_models(*self.args)
                self.result_signal.emit(results)
            elif self.task == "download":
                filepath = self.api.download_model(*self.args)
                self.message_signal.emit("Download Complete", f"Model downloaded to: {filepath}")
            elif self.task == "inference":
                result = self.api.run_inference(*self.args)
                self.inference_result_signal.emit(result)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            self.message_signal.emit(f"{self.task.capitalize()} Error", error_message)
            logging.error(f"Error in {self.task}: {error_message}")