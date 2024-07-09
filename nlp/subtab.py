from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QTabWidget
from nlp import csv_loader_tab
from nlp import word_cloud
from nlp import n_gram
from nlp import kmean
from nlp import hierarchical
from nlp import lda


class SubTab(QWidget):
    def __init__(self, name):
        super().__init__()
        layout = QVBoxLayout()
        label = QLabel(f"This is the content of {name}.")
        layout.addWidget(label)
        self.setLayout(layout)


class NestedTabWidgetS(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        tabWidget = QTabWidget()
        tabWidget.addTab(csv_loader_tab.CSVLoaderTab(), "CSV Loader")
        tabWidget.addTab(word_cloud.WordCloudTab(), "Word Cloud Analysis")
        tabWidget.addTab(n_gram.NGramTab(), "N-Gram Analysis")
        tabWidget.addTab(kmean.KMeansTab(), "K-Means Clustering")
        tabWidget.addTab(
            hierarchical.HierarchicalClusteringTab(), "Hierarchical Clustering"
        )
        tabWidget.addTab(lda.LDATab(), "Latent Dirichlet Allocation")

        layout.addWidget(tabWidget)
        self.setLayout(layout)


class NestedTabWidgetUnS(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        tabWidget = QTabWidget()
        tabWidget.addTab(csv_loader_tab.CSVLoaderTab(), "CSV Loader")
        tabWidget.addTab(word_cloud.WordCloudTab(), "Word Cloud Analysis")
        tabWidget.addTab(n_gram.NGramTab(), "N-Gram Analysis")
        # tabWidget.addTab(kmean.KMeansTab(), "SVM-Linear")
        # tabWidget.addTab(
        #     hierarchical.HierarchicalClusteringTab(), "SVM-RBF"
        # )
        # tabWidget.addTab(lda.LDATab(), "SVM-Polynomial")

        layout.addWidget(tabWidget)
        self.setLayout(layout)
