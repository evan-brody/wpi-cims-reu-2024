from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLineEdit
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QWidget, QPushButton, QLineEdit
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QSplitter, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QDialog,
    QPushButton,
    QLineEdit,
    QMessageBox,
)


class HierarchicalClusteringTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self.dendrogram_button = QPushButton("Generate Dendrogram", self)
        self.dendrogram_button.clicked.connect(self.show_dendrogram_dialog)
        layout.addWidget(self.dendrogram_button)

        self.cutoff_input = QLineEdit(self)
        self.cutoff_input.setPlaceholderText("Enter cutoff distance")
        layout.addWidget(self.cutoff_input)

        self.run_clustering_button = QPushButton("Run Clustering", self)
        self.run_clustering_button.clicked.connect(self.run_clustering)
        layout.addWidget(self.run_clustering_button)

        self.cluster_dropdown = QComboBox(self)
        layout.addWidget(self.cluster_dropdown)
        self.cluster_dropdown.currentIndexChanged.connect(self.display_cluster)

        self.data_table = QTableWidget(self)
        layout.addWidget(self.data_table)

        self.setLayout(layout)

    def show_dendrogram_dialog(self):
        text_data = self.load_processed_text()
        dialog = DendrogramDialog(self, data=text_data)
        dialog.exec_()

    def load_processed_text(self):
        file_path = os.path.join(os.getcwd(), "selected.csv")
        df = pd.read_csv(file_path)
        if "Processed Text" not in df.columns:
            QMessageBox.warning(self, "Error", "Processed Text column is missing.")
            return None
        return df["Processed Text"].dropna().tolist()

    def run_clustering(self):
        cutoff_distance = float(self.cutoff_input.text())
        df = pd.read_csv("selected.csv")
        text_data = df["Processed Text"].dropna().tolist()
        tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        distance_matrix = 1 - similarity_matrix
        linkage_matrix = linkage(distance_matrix, method="average")

        cluster_labels = fcluster(linkage_matrix, cutoff_distance, criterion="distance")
        df["Cluster"] = cluster_labels

        folder_name = "hh_cluster_result"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        else:
            for file in os.listdir(folder_name):
                os.remove(os.path.join(folder_name, file))

        self.cluster_dropdown.clear()
        for cluster_id in np.unique(cluster_labels):
            cluster_data = df[df["Cluster"] == cluster_id]
            cluster_file_path = os.path.join(folder_name, f"hhcluster_{cluster_id}.csv")
            cluster_data.to_csv(cluster_file_path, index=False)
            self.cluster_dropdown.addItem(f"Cluster {cluster_id}", cluster_id)

    def display_cluster(self):
        cluster_id = self.cluster_dropdown.currentData()
        if cluster_id is None:
            return

        folder_name = "hh_cluster_result"
        filename = os.path.join(folder_name, f"hhcluster_{cluster_id}.csv")
        if os.path.exists(filename):
            cluster_data = pd.read_csv(filename)
            self.populate_table(cluster_data)
        else:
            QMessageBox.warning(self, "Error", f"File {filename} does not exist.")

    def populate_table(self, data):
        self.data_table.clear()
        self.data_table.setRowCount(len(data))
        self.data_table.setColumnCount(len(data.columns))
        self.data_table.setHorizontalHeaderLabels(data.columns.tolist())

        for i, (index, row) in enumerate(data.iterrows()):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                self.data_table.setItem(i, j, item)

        self.data_table.resizeColumnsToContents()


class DendrogramDialog(QDialog):
    def __init__(self, parent=None, data=None):
        super().__init__(parent)
        self.setWindowTitle("Hierarchical Clustering Dendrogram")
        self.data = data

        layout = QVBoxLayout(self)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.resize(800, 600)

        self.generate_dendrogram()

    def generate_dendrogram(self):
        tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.data)

        similarity_matrix = cosine_similarity(tfidf_matrix)
        distance_matrix = 1 - similarity_matrix

        linkage_matrix = linkage(distance_matrix, method="average")

        ax = self.figure.add_subplot(111)
        dendrogram(linkage_matrix, ax=ax)
        ax.set_title("Hierarchical Clustering Dendrogram")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Distance")
        self.canvas.draw()
