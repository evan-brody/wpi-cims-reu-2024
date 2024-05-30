import matplotlib as mpl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns

class Charts:
    def __init__(self, main_window):
        self.main_window = main_window

    """
    Refreshes displayed chart with new changes to the table.
    """

    def bar_chart(self):
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
        # Clear the existing plot
        self.main_window.main_figure.clear()

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
        sns.barplot(x="Failure Mode ID", y="RPN", data=df, hue=colors, ax=ax)

        ax.axhline(threshold, color="#68855C", linestyle="--")
        ax.set_ylabel("Risk Priority Number (RPN)")
        ax.set_xlabel("Failure Mode ID")
        component_name = self.main_window.component_name_field.currentText()
        ax.set_title(component_name + " Risk Profile")
        ax.tick_params(axis="x", rotation=0)

        # Set the font to bold
        font = {"weight": "bold"}
        mpl.rc("font", **font)

        # Set the x-axis ticks to integers only
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Refresh the canvas
        self.main_window.canvas.draw()

    """
    Makes a pie chart of data in table.
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

        # Clear the existing plot
        self.main_window.main_figure.clear()

        # Prepare the data for the pie chart
        labels = ["Below Risk Threshold", "Above Risk Threshold"]
        rpn_values = [below_threshold, above_threshold]

        # Set the color of the slices based on the categories
        colors = ["#5f9ea0", "#FF6961"]

        # Create a pie chart
        ax = self.main_window.main_figure.add_subplot(111)
        wedges, texts, autotexts = ax.pie(
            rpn_values, labels=labels, colors=colors, autopct="%1.1f%%", radius=1
        )

        # Create legend
        legend_labels = [
            f"Number of Green Failure Modes: {below_threshold}",
            f"Number of Red Failure Modes: {above_threshold}",
            f"Total Failure Modes: {below_threshold + above_threshold}",
        ]
        ax.legend(
            wedges,
            legend_labels,
            title="Failure Modes",
            loc="upper right",
            bbox_to_anchor=(1, 0.5),
        )

        component_name = self.main_window.component_name_field.currentText()
        ax.set_title(component_name + " Risk Profile")

        # Refresh the canvas
        self.main_window.canvas.draw()

    # def plot_3D(self):
    #     # Copy the plot_3D method code from your main GUI file here

    # def scatterplot(self):
    #     # Copy the scatterplot method code from your main GUI file here

    # def bubble_plot(self):
    #     # Copy the bubble_plot method code from your main GUI file here
