import sys
import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from wordcloud import WordCloud
import pandas as pd


class WordCloudTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)

        self.button = QPushButton("Generate Word Cloud")
        self.button.clicked.connect(self.plot_wordcloud)
        self.layout.addWidget(self.button)

    def plot_wordcloud(self):
        df = pd.read_csv("selected.csv")
        text = " ".join(df["Processed Text"].dropna())

        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
            text
        )

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        self.canvas.draw()

        self.save_wordcloud(wordcloud)

    def save_wordcloud(self, wordcloud):
        folder = "wordcloud_image"
        os.makedirs(folder, exist_ok=True)
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))

        wordcloud.to_file(os.path.join(folder, "wordcloud.png"))
