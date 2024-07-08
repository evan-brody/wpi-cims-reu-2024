import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSpinBox,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from nltk import ngrams
from collections import Counter


class NGramTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)

        self.config_layout = QHBoxLayout()

        self.n_value = QSpinBox(self)
        self.n_value.setRange(1, 5)
        self.config_layout.addWidget(QLabel("N-Value:"))
        self.config_layout.addWidget(self.n_value)

        self.top_n_words = QSpinBox(self)
        self.top_n_words.setRange(1, 50)
        self.config_layout.addWidget(QLabel("Top N Words:"))
        self.config_layout.addWidget(self.top_n_words)

        self.generate_button = QPushButton("Generate N-Gram Graph", self)
        self.generate_button.clicked.connect(self.plot_ngrams)
        self.config_layout.addWidget(self.generate_button)

        self.layout.addLayout(self.config_layout)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)

        self.df = pd.read_csv("selected.csv")

    def plot_ngrams(self):
        n = self.n_value.value()
        top_n = self.top_n_words.value()
        text = " ".join(self.df["Processed Text"].dropna())
        tokens = text.split()

        n_grams = ngrams(tokens, n)
        ngram_freq = Counter(n_grams)
        most_common_ngrams = ngram_freq.most_common(top_n)

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        labels, values = zip(
            *[(" ".join(gram), count) for gram, count in most_common_ngrams]
        )

        ax.barh(labels, values, color="skyblue")
        ax.set_xlabel("Frequency")
        ax.set_title(f"Top {top_n} {n}-grams")
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.invert_yaxis()
        self.figure.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)

        self.canvas.draw()
