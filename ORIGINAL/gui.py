from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLineEdit, QPushButton, QListWidget, QLabel, 
                             QComboBox, QFileDialog, QProgressBar, QMessageBox,
                             QTabWidget, QFormLayout, QTextEdit, QGroupBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPalette, QColor

class MainWindow(QMainWindow):
    theme_signal = pyqtSignal(str)
    search_signal = pyqtSignal(str, dict)
    download_signal = pyqtSignal(str, str)
    api_key_signal = pyqtSignal(str)
    default_dir_signal = pyqtSignal(str)
    inference_signal = pyqtSignal(str, str)

    def __init__(self):
        self.current_theme = "light"
        super().__init__()
        self.setWindowTitle("Hugging Face Hub Explorer")
        self.setGeometry(100, 100, 1000, 700)
        self.setup_style()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.tabs = QTabWidget(self.central_widget)
        self.layout.addWidget(self.tabs)

        self.setup_search_tab()
        self.setup_settings_tab()
        self.setup_inference_tab()

    def setup_style(self):
        if self.current_theme == "light":
            self.setStyleSheet("""
                QMainWindow { background-color: #f0f0f0; }
                QTabWidget::pane { border: 1px solid #cccccc; background-color: #ffffff; color: #000000; }
                QTabBar::tab { background-color: #e0e0e0; padding: 8px 16px; margin-right: 2px; }
                QTabBar::tab:selected { background-color: #ffffff; border-bottom: 2px solid #4a86e8; }
                QPushButton { background-color: #4a86e8; color: white; padding: 8px 16px; border: none; border-radius: 4px; }
                QPushButton:hover { background-color: #3a76d8; }
                QLineEdit, QComboBox { padding: 6px; border: 1px solid #cccccc; border-radius: 4px; }
                QListWidget { border: 1px solid #cccccc; border-radius: 4px; }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow { background-color: #2e2e2e; }
                QTabWidget::pane { border: 1px solid #444444; background-color: #3c3c3c; color: #e0e0e0; }
                QTabBar::tab { background-color: #4a4a4a; padding: 8px 16px; margin-right: 2px; color: #e0e0e0; }
                QTabBar::tab:selected { background-color: #3c3c3c; border-bottom: 2px solid #76a9ea; }
                QPushButton { background-color: #76a9ea; color: white; padding: 8px 16px; border: none; border-radius: 4px; }
                QPushButton:hover { background-color: #6598d8; }
                QLineEdit, QComboBox { padding: 6px; border: 1px solid #444444; border-radius: 4px; color: #e0e0e0; background-color: #2e2e2e; }
                QListWidget { border: 1px solid #444444; border-radius: 4px; color: #e0e0e0; background-color: #2e2e2e; }
                QLabel { color: #e0e0e0; }
                QTextEdit { color: #e0e0e0; background-color: #2e2e2e; border: 1px solid #444444; }
            """)

    def change_theme(self, index):
        self.current_theme = "dark" if index == 1 else "light"
        self.setup_style()
        self.theme_signal.emit(self.current_theme)

    def setup_search_tab(self):
        search_tab = QWidget()
        search_layout = QVBoxLayout(search_tab)

        search_bar_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search models...")
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.on_search)
        search_bar_layout.addWidget(self.search_input)
        search_bar_layout.addWidget(self.search_button)
        search_layout.addLayout(search_bar_layout)

        filter_layout = QHBoxLayout()
        self.task_filter = QComboBox()
        self.task_filter.addItems(["All Tasks", "Text Classification", "Image Generation", "Translation"])
        self.library_filter = QComboBox()
        self.library_filter.addItems(["All Libraries", "PyTorch", "TensorFlow", "JAX"])
        filter_layout.addWidget(QLabel("Task:"))
        filter_layout.addWidget(self.task_filter)
        filter_layout.addWidget(QLabel("Library:"))
        filter_layout.addWidget(self.library_filter)
        search_layout.addLayout(filter_layout)

        self.results_list = QListWidget()
        search_layout.addWidget(self.results_list)

        download_layout = QHBoxLayout()
        self.download_button = QPushButton("Download Selected")
        self.download_button.clicked.connect(self.on_download)
        self.progress_bar = QProgressBar()
        download_layout.addWidget(self.download_button)
        download_layout.addWidget(self.progress_bar)
        search_layout.addLayout(download_layout)

        self.tabs.addTab(search_tab, "Search")

    def setup_settings_tab(self):
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)

        api_key_group = QGroupBox("API Key")
        api_key_layout = QFormLayout()
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_save = QPushButton("Save API Key")
        self.api_key_save.clicked.connect(self.save_api_key)
        api_key_layout.addRow("API Key:", self.api_key_input)
        api_key_layout.addRow(self.api_key_save)
        api_key_group.setLayout(api_key_layout)
        settings_layout.addWidget(api_key_group)

        download_dir_group = QGroupBox("Default Download Directory")
        download_dir_layout = QFormLayout()
        self.download_dir_input = QLineEdit()
        self.download_dir_browse = QPushButton("Browse")
        self.download_dir_browse.clicked.connect(self.browse_download_dir)
        self.download_dir_save = QPushButton("Save Directory")
        self.download_dir_save.clicked.connect(self.save_download_dir)
        download_dir_layout.addRow("Directory:", self.download_dir_input)
        download_dir_layout.addRow(self.download_dir_browse)
        download_dir_layout.addRow(self.download_dir_save)
        download_dir_group.setLayout(download_dir_layout)
        settings_layout.addWidget(download_dir_group)

        theme_group = QGroupBox("Theme")
        theme_layout = QFormLayout()
        self.theme_selector = QComboBox()
        self.theme_selector.addItems(["Light", "Dark"])
        self.theme_selector.currentIndexChanged.connect(self.change_theme)
        theme_layout.addRow("Select Theme:", self.theme_selector)
        theme_group.setLayout(theme_layout)
        settings_layout.addWidget(theme_group)

        settings_layout.addStretch(1)
        self.tabs.addTab(settings_tab, "Settings")
        
    def setup_inference_tab(self):
        inference_tab = QWidget()
        inference_layout = QVBoxLayout(inference_tab)

        model_layout = QHBoxLayout()
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("Enter model ID")
        self.model_input.setToolTip("Enter the full model name, e.g., 'microsoft/phi-3.5-medium'")
        model_layout.addWidget(QLabel("Model ID:"))
        model_layout.addWidget(self.model_input)
        inference_layout.addLayout(model_layout)

        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Enter input text or data")
        self.input_text.setToolTip("Enter the text you want to process with the model")
        inference_layout.addWidget(QLabel("Input:"))
        inference_layout.addWidget(self.input_text)

        self.run_inference_button = QPushButton("Run Inference")
        self.run_inference_button.clicked.connect(self.on_run_inference)
        inference_layout.addWidget(self.run_inference_button)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        inference_layout.addWidget(QLabel("Output:"))
        inference_layout.addWidget(self.output_text)

        self.tabs.addTab(inference_tab, "Inference Playground")    

    def on_search(self):
        query = self.search_input.text()
        filters = {
            "task": self.task_filter.currentText() if self.task_filter.currentText() != "All Tasks" else None,
            "library": self.library_filter.currentText() if self.library_filter.currentText() != "All Libraries" else None
        }
        self.search_signal.emit(query, filters)

    def on_download(self):
        selected_items = self.results_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a model to download.")
            return
        model_id = selected_items[0].text()
        download_dir = self.download_dir_input.text() or QFileDialog.getExistingDirectory(self, "Select Download Directory")
        if download_dir:
            self.download_signal.emit(model_id, download_dir)

    def save_api_key(self):
        api_key = self.api_key_input.text()
        self.api_key_signal.emit(api_key)
        QMessageBox.information(self, "API Key Saved", "Your API key has been saved.")

    def browse_download_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Default Download Directory")
        if dir_path:
            self.download_dir_input.setText(dir_path)

    def save_download_dir(self):
        download_dir = self.download_dir_input.text()
        self.default_dir_signal.emit(download_dir)
        QMessageBox.information(self, "Directory Saved", "Your default download directory has been saved.")
        
    def on_run_inference(self):
        model_id = self.model_input.text()
        input_data = self.input_text.toPlainText()
        self.inference_signal.emit(model_id, input_data)    

    def update_results(self, results):
        self.results_list.clear()
        sorted_results = sorted(results, key=lambda x: x['id'].lower())
        for result in sorted_results:
            self.results_list.addItem(result['id'])

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def show_message(self, title, message):
        QMessageBox.information(self, title, message)
    
    def update_inference_output(self, result):
        self.output_text.setPlainText(str(result))