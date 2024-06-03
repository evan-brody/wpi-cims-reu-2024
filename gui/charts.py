import matplotlib as mpl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
from PyQt5.QtWidgets import QMessageBox


class Charts:
    def __init__(self, main_window):
        self.main_window = main_window

    """
    Refreshes displayed chart with new changes to the table.
    """

    def bar_chart(self):
        # Clear the existing plot
        self.main_window.main_figure.clear()

        component_data = []
        threshold = float(self.main_window.threshold_field.text())

        for row in range(self.main_window.table_widget.rowCount()):
            failure_mode_item = self.main_window.table_widget.item(row, 0)
            rpn_item = self.main_window.table_widget.item(row, 1)
            if failure_mode_item and rpn_item:
                component_data.append(
                    {
                        "id": int(row),
                        "failure_mode": failure_mode_item.text(),
                        "rpn": float(rpn_item.text()),
                    }
                )

        # Adjust the subplot for spacing
        self.main_window.main_figure.subplots_adjust(
            left=0.18
        )  # You can adjust the value to suit your needs

        # Extract the failure modes and RPN values
        ids = [data["id"] for data in component_data]
        rpn_values = [data["rpn"] for data in component_data]

        # Convert the IDs to integers
        ids = list(map(int, ids))

        # Create a DataFrame for seaborn
        df = pd.DataFrame({"Failure Mode ID": ids, "RPN": rpn_values})

        # Create a bar plot
        ax = self.main_window.main_figure.add_subplot(111)

        # Set the color of the bars based on RPN values
        colors = ["#5f9ea0" if rpn < threshold else "#FF6961" for rpn in rpn_values]
        df["Threshold"] = [
            "Below Threshold" if rpn < threshold else "Above Threshold"
            for rpn in rpn_values
        ]
        sns.barplot(
            x="Failure Mode ID",
            y="RPN",
            data=df,
            hue="Threshold",
            ax=ax,
            palette={"Below Threshold": "#5f9ea0", "Above Threshold": "#FF6961"},
        )
        handles, labels = ax.get_legend_handles_labels()
        labels = [
            "Below Threshold" if label == "Below Threshold" else "Above Threshold"
            for label in labels
        ]
        ax.legend(handles, labels)

        ax.axhline(threshold, color="#68855C", linestyle="--")
        ax.set_ylabel("Risk Priority Number (RPN)")
        ax.set_xlabel("Failure Mode ID")
        component_name = self.main_window.component_name_field.currentText()
        ax.set_title(component_name + " RPN Bar Chart")
        ax.tick_params(axis="x", rotation=0)

        # Set the font to bold
        font = {"weight": "bold"}
        mpl.rc("font", **font)

        # Set the x-axis ticks to integers only
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Refresh the canvas
        self.main_window.canvas.draw()

    """
    Displays a pie chart of data in table.
    """

    def pie_chart(self):
        # Clear the existing plot
        self.main_window.main_figure.clear()

        component_data = []
        threshold = float(self.main_window.threshold_field.text())
        below_threshold = 0
        above_threshold = 0

        for row in range(self.main_window.table_widget.rowCount()):
            failure_mode_item = self.main_window.table_widget.item(row, 1)
            rpn_item = self.main_window.table_widget.item(row, 2)
            if failure_mode_item and rpn_item:
                rpn = float(rpn_item.text())
                if rpn < threshold:
                    below_threshold += 1
                else:
                    above_threshold += 1

        # Prepare the data for the pie chart
        labels = ["Below Risk Threshold", "Above Risk Threshold"]
        rpn_values = [below_threshold, above_threshold]

        # Set the color of the slices based on the categories
        colors = ["#5f9ea0", "#FF6961"]

        # Create a pie chart
        ax = self.main_window.main_figure.add_subplot(111)
        wedges, texts, autotexts = ax.pie(
            rpn_values, labels=labels, colors=colors, autopct="%1.1f%%", radius=0.7
        )

        # Directly label the numbers on the pie chart
        total_failure_modes = below_threshold + above_threshold
        labels = [
            f"{below_threshold} ({below_threshold / total_failure_modes:.1%})",
            f"{above_threshold} ({above_threshold / total_failure_modes:.1%})",
        ]
        for i, autotext in enumerate(autotexts):
            autotext.set_text(labels[i])
            autotext.set_fontsize(10)  # Adjust the font size as needed

        for text in texts:
            text.set_fontsize(10)

        component_name = self.main_window.component_name_field.currentText()
        ax.set_title(component_name + " Risk Profile")

        # Refresh the canvas
        self.main_window.canvas.draw()

    """
    Displays 3D plot of data in table.
    Takes in 2 element array with x and y coordinates of cell
    """

    def plot_3D(self, cell_location):
        # Clear the existing plot
        self.main_window.main_figure.clear()
        length = float(self.main_window.table_widget.item(cell_location[0], 2).text())
        width = float(self.main_window.table_widget.item(cell_location[0], 3).text())
        height = float(self.main_window.table_widget.item(cell_location[0], 4).text())

        # Get the X, Y, and Z values

        """
        try:
            length = float(self.main_window.x_input_field.text())
            width = float(self.main_window.y_input_field.text())
            height = float(self.main_window.z_input_field.text())
        except ValueError:
            QMessageBox.critical(
                self.main_window,
                "Value Error",
                "Please enter valid numbers for Frequency, Severity, and Detection.",
            )
            return
        """

        # Calculate RPN
        rpn = length * width * height

        # Determine color
        if rpn < 560:
            color = "green"
        elif 560 <= rpn < 840:
            color = "yellow"
        else:
            color = "red"

        # Generate the surface plot
        self.main_window.main_figure.clear()
        ax = self.main_window.main_figure.add_subplot(111, projection="3d")

        # Create a list of 3D coordinates for the vertices of each face
        vertices = [
            [
                (0, 0, 0),
                (0, width, 0),
                (length, width, 0),
                (length, 0, 0),
            ],  # Bottom face
            [
                (0, 0, 0),
                (0, 0, height),
                (length, 0, height),
                (length, 0, 0),
            ],  # Front face
            [(0, 0, 0), (0, 0, height), (0, width, height), (0, width, 0)],  # Left face
            [
                (length, 0, 0),
                (length, 0, height),
                (length, width, height),
                (length, width, 0),
            ],  # Right face
            [
                (0, 0, height),
                (0, width, height),
                (length, width, height),
                (length, 0, height),
            ],  # Top face
            [
                (0, width, 0),
                (0, width, height),
                (length, width, height),
                (length, width, 0),
            ],  # Rear face
        ]

        # Add the faces to the plot
        for face in vertices:
            ax.add_collection3d(
                Poly3DCollection(
                    [face], alpha=0.25, linewidths=1, edgecolors="r", facecolors=color
                )
            )

        ax.set_xlabel("Frequency")
        ax.set_ylabel("Severity")
        ax.set_zlabel("Detection")

        ax.set_xlim([0, length])
        ax.set_ylim([0, width])
        ax.set_zlim([0, height])

        self.main_window.canvas.draw()

    """
    Displays a 3D scatterplot of data (Frequency, Severity, Detection) in table.
    """

    def scatterplot(self):
        component_data = []
        threshold = float(self.main_window.threshold_field.text())

        for row in range(self.main_window.table_widget.rowCount()):
            frequency_item = self.main_window.table_widget.item(row, 2)
            severity_item = self.main_window.table_widget.item(row, 3)
            detection_item = self.main_window.table_widget.item(row, 4)
            if severity_item and detection_item and frequency_item:
                component_data.append(
                    {
                        "id": int(row),
                        "severity": float(severity_item.text()),
                        "detection": float(detection_item.text()),
                        "frequency": float(frequency_item.text()),
                    }
                )

        # Clear the existing plot
        self.main_window.main_figure.clear()

        # Extract the values
        ids = [data["id"] for data in component_data]
        severity_values = [data["severity"] for data in component_data]
        detection_values = [data["detection"] for data in component_data]
        frequency_values = [data["frequency"] for data in component_data]

        df = pd.DataFrame(
            {
                "Failure Mode ID": ids,
                "Severity": severity_values,
                "Detection": detection_values,
                "Frequency": frequency_values,
            }
        )

        # Create a 3D scatterplot
        ax = self.main_window.main_figure.add_subplot(111, projection="3d")

        sc = ax.scatter(
            df["Severity"],
            df["Detection"],
            df["Frequency"],
            c=df["Failure Mode ID"],
            cmap="viridis",
        )

        ax.set_xlabel("Severity")
        ax.set_ylabel("Detection")
        ax.set_zlabel("Frequency")
        component_name = self.main_window.component_name_field.currentText()
        ax.set_title(component_name + " Risk Profile")

        # Add a colorbar
        self.main_window.main_figure.colorbar(sc, ax=ax, pad=0.02)

        # Refresh the canvas
        self.main_window.canvas.draw()

    """
    Makes a bubble chart of data in table. Builds upon the scatterplot function by altering bubbles to size according to RPN
    """

    def bubble_plot(self):
        component_data = []
        threshold = float(self.main_window.threshold_field.text())

        for row in range(self.main_window.table_widget.rowCount()):
            frequency_item = self.main_window.table_widget.item(row, 2)
            severity_item = self.main_window.table_widget.item(row, 3)
            detection_item = self.main_window.table_widget.item(row, 4)
            if severity_item and detection_item and frequency_item:
                component_data.append(
                    {
                        "id": int(row),
                        "severity": float(severity_item.text()),
                        "detection": float(detection_item.text()),
                        "frequency": float(frequency_item.text()),
                    }
                )

        ids = [data["id"] for data in component_data]
        severity_values = [data["severity"] for data in component_data]
        detection_values = [data["detection"] for data in component_data]
        frequency_values = [data["frequency"] for data in component_data]

        rpn_values = [
            data["severity"] * data["detection"] * data["frequency"]
            for data in component_data
        ]

        rpn_scaled = [
            np.cbrt(val) * 30 for val in rpn_values
        ]  # Adjust scaling factor as needed

        # Create a 3D plot
        self.main_window.main_figure.clear()
        ax = self.main_window.main_figure.add_subplot(111, projection="3d")

        bubble = ax.scatter(
            frequency_values,
            severity_values,
            detection_values,
            s=rpn_scaled,
            c=rpn_scaled,
            cmap="viridis",
            edgecolors="black",
            alpha=0.6,
        )

        # Adding titles and labels
        component_name = self.main_window.component_name_field.currentText()
        ax.set_title(component_name + " 3D Bubble Plot")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Severity")
        ax.set_zlabel("Detection")

        # Color bar which maps values to colors.
        cbar = self.main_window.main_figure.colorbar(
            bubble, ax=ax, shrink=0.5, aspect=5
        )
        cbar.set_label("Risk Priority Number (RPN)")

        # plt.show()
        self.main_window.canvas.draw()
