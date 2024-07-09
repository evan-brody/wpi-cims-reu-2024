import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QLineEdit,
    QMessageBox,
    QDialog,
    QFormLayout,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class LDATab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self.loglike_perplex_button = QPushButton(
            "Generate Log-Likehood and Perplexity Graph", self
        )
        self.loglike_perplex_button.clicked.connect(self.handle_button_click)
        self.layout().addWidget(self.loglike_perplex_button)

        self.k_input = QLineEdit(self)
        self.k_input.setPlaceholderText("Enter number of topics")
        layout.addWidget(self.k_input)

        self.run_button = QPushButton("Run LDA", self)
        self.run_button.clicked.connect(self.run_lda)
        layout.addWidget(self.run_button)

        self.cluster_dropdown = QComboBox(self)
        layout.addWidget(self.cluster_dropdown)
        self.cluster_dropdown.currentIndexChanged.connect(self.display_cluster)

        self.data_table = QTableWidget(self)
        layout.addWidget(self.data_table)

        self.setLayout(layout)

    def load_processed_text(self):
        file_path = os.path.join(os.getcwd(), "selected.csv")
        df = pd.read_csv(file_path)
        if "Processed Text" not in df.columns:
            raise ValueError("Processed Text column is missing in the CSV file.")
        return df["Processed Text"].dropna().tolist()

    def run_lda(self):
        k = int(self.k_input.text())
        df = pd.read_csv("selected.csv")
        if "Processed Text" not in df.columns:
            QMessageBox.warning(self, "Error", "Processed Text column is missing.")
            return

        text_data = df["Processed Text"].dropna().tolist()

        if not text_data:
            QMessageBox.warning(self, "Error", "No processed text data available.")
            return

        lda = LatentDirichletAllocation(n_components=k, max_iter=10, random_state=100)
        vectorizer = CountVectorizer(max_df=0.5, min_df=1, stop_words="english")
        data_vectorized = vectorizer.fit_transform(text_data)
        lda.fit(data_vectorized)

        doc_topic_distr = lda.transform(data_vectorized)
        doc_topic_labels = doc_topic_distr.argmax(axis=1)
        labels = doc_topic_labels

        folder_name = "lda_result"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        else:
            for file in os.listdir(folder_name):
                os.remove(os.path.join(folder_name, file))

        self.cluster_dropdown.clear()
        for i in range(k):
            cluster_data = df.iloc[labels == i]

            cluster_file_path = os.path.join(folder_name, f"cluster_{i}.csv")
            cluster_data.to_csv(cluster_file_path, index=False)
            self.cluster_dropdown.addItem(f"Cluster {i}")

        # data["Assigned Topic"] = doc_topic_labels

        # folder_name = "lda_result"
        # if not os.path.exists(folder_name):
        #     os.makedirs(folder_name)
        # else:
        #     for file in os.listdir(folder_name):
        #         os.remove(os.path.join(folder_name, file))

        # # tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        # # tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)

        # # kmeans = KMeans(n_clusters=k, random_state=42)
        # # kmeans.fit(tfidf_matrix)
        # # labels = kmeans.labels_

        # folder_name = "k_cluster_result"
        # if not os.path.exists(folder_name):
        #     os.makedirs(folder_name)
        # else:
        #     for file in os.listdir(folder_name):
        #         os.remove(os.path.join(folder_name, file))

        # self.cluster_dropdown.clear()
        # for i in range(k):
        #     cluster_data = df.iloc[labels == i]

        #     cluster_file_path = os.path.join(folder_name, f"cluster_{i}.csv")
        #     cluster_data.to_csv(cluster_file_path, index=False)
        #     self.cluster_dropdown.addItem(f"Cluster {i}")

    def handle_button_click(self):
        text_data = self.load_processed_text()
        dialog = LDADialog(self, data=text_data)
        dialog.exec_()

    def display_cluster(self, index):
        folder_name = "lda_result"
        filename = os.path.join(folder_name, f"cluster_{index}.csv")
        if os.path.exists(filename):
            cluster_data = pd.read_csv(filename)
            self.populate_table(cluster_data)
        else:
            QMessageBox.warning(self, "Error", f"File {filename} does not exist.")

    def populate_table(self, data):
        self.data_table.setRowCount(data.shape[0])
        self.data_table.setColumnCount(data.shape[1])
        self.data_table.setHorizontalHeaderLabels(data.columns.tolist())

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                self.data_table.setItem(i, j, QTableWidgetItem(str(data.iloc[i, j])))

        self.data_table.resizeColumnsToContents()


class LDADialog(QDialog):
    def __init__(self, parent=None, data=None):
        super().__init__(parent)
        self.setWindowTitle("Select Range for number of topics")
        self.data = data

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.min_k_input = QLineEdit(self)
        self.max_k_input = QLineEdit(self)

        form_layout.addRow("Min:", self.min_k_input)
        form_layout.addRow("Max:", self.max_k_input)
        layout.addLayout(form_layout)

        self.plot_button = QPushButton("Generate Plot", self)
        self.plot_button.clicked.connect(self.plot_method)
        layout.addWidget(self.plot_button)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.close)
        layout.addWidget(self.close_button)

    def plot_method(self):
        min_k = int(self.min_k_input.text())
        max_k = int(self.max_k_input.text())
        if min_k >= max_k or min_k < 1:
            QMessageBox.warning(
                self, "Invalid Input", "Please ensure min k < max k and min k >= 1."
            )
            return

        vectorizer = CountVectorizer(max_df=0.5, min_df=1, stop_words="english")
        data_vectorized = vectorizer.fit_transform(self.data)
        n_topics_array = np.arange(min_k, max_k + 1)
        log_likelihoods = []
        perplexities = []

        for n_topics in n_topics_array:
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=10,
                learning_method="online",
                random_state=100,
            )
            lda.fit(data_vectorized)
            log_likelihoods.append(lda.score(data_vectorized))
            perplexities.append(lda.perplexity(data_vectorized))

        self.figure.clear()
        ax1 = self.figure.add_subplot(111)
        ax1.plot(
            n_topics_array,
            log_likelihoods,
            marker="o",
            label="Log-Likelihood",
            color="tab:red",
        )
        ax1.set_xlabel("Number of Topics")
        ax1.set_ylabel("Log Likelihood", color="tab:red")
        ax1.set_title("Log Likelihood for LDA")
        ax1.tick_params(axis="y", labelcolor="tab:red")

        ax2 = ax1.twinx()
        ax2.plot(
            n_topics_array,
            perplexities,
            marker="x",
            label="Perplexity",
            color="tab:blue",
        )
        ax2.set_ylabel("Perplexity", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

        self.canvas.draw()
