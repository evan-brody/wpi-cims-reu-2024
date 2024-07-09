import sys
import pandas as pd
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
    QTableWidget,
    QTableWidgetItem,
    QGridLayout,
    QSpinBox,
)
from PyQt5.QtGui import QFontMetrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityAnalysisTab(QWidget):
    def __init__(self):
        super().__init__()
        self.data = self.load_processed_text()  # Load your dataset using a method
        self.layout = QVBoxLayout(self)

        # Initialize UI components
        self.setup_user_inputs()
        self.setup_results_table()

    def load_processed_text(self):
        """Method to load and preprocess the dataset."""
        file_path = "fmeca_data.csv"
        df = pd.read_csv(file_path, header=0)
        df.fillna("", inplace=True)  # Handle NaN values for text processing
        return df

    def setup_user_inputs(self):
        """Setup the UI for user inputs."""
        self.grid_layout = QGridLayout()
        self.field_selectors = {}
        self.description_inputs = {}
        fields = [
            "Function",
            "Failure Cause",
            "Local Failure Effect",
            "Next Higher Effect",
            "End Effect",
            "Detection Method",
            "Compensating Provisions",
        ]

        # Create input fields and selectors dynamically
        for i, field in enumerate(fields):
            label = QLabel(field)
            combo_box = QComboBox()
            combo_box.addItems(["Include", "Exclude"])
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(f"Enter description for {field}")

            self.field_selectors[field] = combo_box
            self.description_inputs[field] = line_edit

            self.grid_layout.addWidget(label, i, 0)
            self.grid_layout.addWidget(combo_box, i, 1)
            self.grid_layout.addWidget(line_edit, i, 2)

        # SpinBox for selecting number of top results
        self.top_results_label = QLabel("Number of top results:")
        self.top_results_spinbox = QSpinBox()
        self.top_results_spinbox.setMinimum(1)
        self.top_results_spinbox.setMaximum(50)
        self.top_results_spinbox.setValue(10)  # Default to showing top 10 results
        self.grid_layout.addWidget(self.top_results_label, len(fields), 0)
        self.grid_layout.addWidget(self.top_results_spinbox, len(fields), 1)

        self.analyze_button = QPushButton("Analyze Similarity")
        self.analyze_button.clicked.connect(self.perform_analysis)
        self.layout.addLayout(self.grid_layout)
        self.layout.addWidget(self.analyze_button)

    def setup_results_table(self):
        """Setup the results display table."""
        self.results_table = QTableWidget()
        headers = self.data.columns.tolist() + ["Similarity Score"]
        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        self.layout.addWidget(self.results_table)

        # Adjust column widths based on header lengths
        header_font = QFontMetrics(self.results_table.horizontalHeader().font())
        for i, header in enumerate(headers):
            self.results_table.setColumnWidth(i, header_font.width(header) + 20)

    def perform_analysis(self):
        """Perform the similarity analysis based on user's input and update the results table."""
        descriptions = {
            field: line_edit.text()
            for field, line_edit in self.description_inputs.items()
            if self.field_selectors[field].currentText() == "Include"
            and line_edit.text().strip()
        }

        if descriptions:
            top_n = self.top_results_spinbox.value()
            similar_indices, scores = self.find_similar_fmeca_id_stepwise(
                descriptions, top_n
            )
            self.update_results_table(similar_indices, scores)
        else:
            QMessageBox.warning(
                self,
                "Input Error",
                "Please include at least one field with a description.",
            )

    def find_similar_fmeca_id_stepwise(self, descriptions, top_n):
        """Calculate similarity scores for the entered descriptions against the dataset."""
        vectorizers = {field: TfidfVectorizer() for field in descriptions}
        similarity_scores = pd.DataFrame(index=self.data.index)

        for field, description in descriptions.items():
            combined_texts = self.data[field].tolist() + [description]
            vectorized_texts = vectorizers[field].fit_transform(combined_texts)
            cosine_sim = cosine_similarity(
                vectorized_texts[-1], vectorized_texts[:-1]
            ).flatten()
            similarity_scores[field] = cosine_sim

        similarity_scores["average"] = similarity_scores.mean(axis=1)
        top_indices = similarity_scores["average"].nlargest(top_n).index
        return top_indices, similarity_scores.loc[top_indices, "average"]

    # def update_results_table(self, indices, scores):
    #     """Update the results table with new data."""
    #     self.results_table.setRowCount(len(indices))
    #     for i, idx in enumerate(indices):
    #         for j, col in enumerate(self.data.columns):
    #             self.results_table.setItem(
    #                 i, j, QTableWidgetItem(str(self.data.iloc[idx, j]))
    #             )
    #         self.results_table.setItem(
    #             i, len(self.data.columns), QTableWidgetItem(f"{scores.iloc[idx]:.4f}")
    #         )
    def update_results_table(self, indices, scores):
        """Update the results table with new data."""
        self.results_table.setRowCount(len(indices))
        for i, idx in enumerate(indices):
            for j, col in enumerate(self.data.columns):
                self.results_table.setItem(
                    i, j, QTableWidgetItem(str(self.data.iloc[idx, j]))
                )
            # Access the scores directly using the index from 'indices'
            self.results_table.setItem(
                i, len(self.data.columns), QTableWidgetItem(f"{scores[idx]:.4f}")
            )
