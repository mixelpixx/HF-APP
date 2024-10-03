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

        self.status_label = QLabel("Ready")
        search_layout.addWidget(self.status_label)

        self.results_list = QListWidget()
        search_layout.addWidget(self.results_list)