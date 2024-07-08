import pandas as pd
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QFileDialog,
    QMessageBox,
    QHeaderView,
)
import os
from nlp import preprocess


class CSVLoaderTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self.loadButton = QPushButton("Load CSV")
        self.loadButton.clicked.connect(self.load_csv)
        layout.addWidget(self.loadButton)

        self.columnSelector = QComboBox()
        self.columnSelector.currentIndexChanged.connect(self.display_column_data)
        layout.addWidget(self.columnSelector)

        self.dataTable = QTableWidget()
        layout.addWidget(self.dataTable)

        self.saveButton = QPushButton("Save Selected Column")
        self.saveButton.clicked.connect(self.save_selected_column)
        layout.addWidget(self.saveButton)

        self.preprocessButton = QPushButton("Preprocess Text")
        self.preprocessButton.clicked.connect(self.preprocess_text)
        layout.addWidget(self.preprocessButton)

        self.setLayout(layout)
        self.df = None

    def load_csv(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv)"
        )
        if file_name:
            self.df = pd.read_csv(file_name)
            self.columnSelector.clear()
            self.columnSelector.addItems(self.df.columns)

    def display_column_data(self):
        selected_column = self.columnSelector.currentText()
        if self.df is not None and selected_column:
            self.dataTable.clear()
            self.dataTable.setRowCount(len(self.df[selected_column]))
            self.dataTable.setColumnCount(1)
            self.dataTable.setHorizontalHeaderLabels([selected_column])
            self.dataTable.horizontalHeader().setStretchLastSection(True)
            for row_number, value in enumerate(self.df[selected_column]):
                self.dataTable.setItem(row_number, 0, QTableWidgetItem(str(value)))
            self.dataTable.setSelectionBehavior(QTableWidget.SelectRows)

    def save_selected_column(self):
        if self.df is not None and self.columnSelector.currentText():
            column_name = self.columnSelector.currentText()
            save_path = os.path.join(os.getcwd(), "selected.csv")
            try:

                self.df[[column_name]].to_csv(save_path, index=False)

                msg = QMessageBox()
                msg.setWindowTitle("File Saved")
                msg.setText(
                    "'selected.csv' was successfully saved in the current directory."
                )
                msg.setIcon(QMessageBox.Information)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()

            except Exception as e:
                msg = QMessageBox()
                msg.setWindowTitle("Error")
                msg.setText("Failed to save the file:\n" + str(e))
                msg.setIcon(QMessageBox.Critical)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()

    def preprocess_text(self):
        file_path = os.path.join(os.getcwd(), "selected.csv")
        try:
            df = pd.read_csv(file_path)
            column_name = (
                self.columnSelector.currentText()
                if self.columnSelector.currentText()
                else "Column"
            )

            df["Processed Text"] = df[column_name].apply(
                lambda x: " ".join(preprocess.preprocess_text(x))
            )

            df.to_csv(file_path, index=False)

            self.update_display(df, column_name)

            QMessageBox.information(
                self, "Process Complete", "Text has been processed and saved."
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                "Failed to process the text:\n"
                + str(e)
                + "\nPlease load and save a CSV file first.",
            )

    def update_display(self, df, column_name):
        self.dataTable.clear()
        self.dataTable.setRowCount(len(df))
        self.dataTable.setColumnCount(2)
        self.dataTable.setHorizontalHeaderLabels([column_name, "Processed Text"])

        for row_number in range(len(df)):
            self.dataTable.setItem(
                row_number, 0, QTableWidgetItem(str(df[column_name].iloc[row_number]))
            )
            self.dataTable.setItem(
                row_number,
                1,
                QTableWidgetItem(str(df["Processed Text"].iloc[row_number])),
            )

        self.dataTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.dataTable.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
