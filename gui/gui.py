"""

Project: AI-Based FMEA Tool
2023 Contributor(s): Nicholas Grabill, Stephanie Wang
2024 Contributor(s): Evan Brody, Karl Ramus, Esther Yu
Description: Blah, blah. Blah.
Industry Mentor(s): Dr. Yehia Khalil
Academic Mentor(s): Dr. Frank Zou, Dr. Tharindu DeAlwis, Hammed Olayinka

File(s):
1) gui.py
    1a) Runs GUI implemented in Python w/ PyQt5 framework
2) statistics.py
    2a) Statistical modeling and analysis w/ framework described in paper
3) database.csv
    3a) Holds data for the GUI
4) humanoid.csv
    4a) Holds data for the "Expert Humanoid in the Loop"

NSF REU Project under grant DMS-2244306 in Industrial Mathematics & Statistics sponsored by Collins Aerospace and
Worcester Polytechnic Institute.

### TASK LIST ###

TODO: Refactor the code so that it can generate Weibull/Rayleigh plots for any component using defaults if data isn't
      provided. (WIP, this will take more doing)
TODO: Add bathtub curve in Statistics tab with options for parameter changes. (WIP, just need to write some code)
DONE: RPN & FSD should be Ints

TODO: UI Bug fixes
    DONE: Not putting in risk acceptance threshold makes main tool crash when try to generate
    TODO: Tables are same size so labels are cut off
    DONE: generating weibull distribution in stats tab crashes
    DONE: generating rayleigh distribution in stats tab crashes
    DONE: generate plot in stats distribution crashes app unless you modify something first
    TODO: Source Plot1,2,3 from database instead of hardcoded
    TODO: Variable plot sizes in stats tab
    DONE: Download chart button downloads blank jpeg
    TODO: Read database pulls from csv not local storage/current modified database
    TODO: Save RPN values should save it to a file not just locally
    TODO: modifying FSD variables should auto save to local database instead of having button do it
    TODO: modifying FSD variables should wait till you leave the text box to throw out of bounds errors
    TODO: all data modification should just be in the table, no need for textboxes
    TODO: FMEA and FMECA buttons exist but don't do anything 
    TODO: risk acceptance should autocolor when table is generated
    TODO: detectability recommendation should reset when selected component is changed
    TODO: read database at startup
    TODO: automatically update database
    DONE: stats show table without selecting crashes
    DONE: synced component select between tabs
    

DONE: bubbleplot should open to app and not browser
TODO: Search for components
DONE: generate_chart() if block to switch case
TODO: values() design fix, also figure out what it does ???
DONE: fix csv formatting. there shouldn't be spaces after commas
DONE: convert .csv to sqlite .db file
DONE: normalize database
TODO: re-implement dictionary database as pandas dataframe, populated from SQLite database

"""

import os, sys, csv, sqlite3, stats
import seaborn as sns
import matplotlib as mpl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

database_data = {}

DEFAULT_RISK_THRESHOLD = 1

"""

Name: MainWindow
Type: class
Description: MainWindow class that holds all of our functions for the GUI.

"""


class MainWindow(QMainWindow):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(os.path.dirname(__file__), "..", "data")
    default_db_name = "part_info.db"
    recommendations = [
        "Recommended Detectability: 9-10 (Unacceptable)",
        "Recommended Detectability: 7-8 (Severe)",
        "Recommended Detectability: 4-6 (Medium)",
        "Recommended Detectability: 1-3 (Low)"
        ]
    # Columns to show in the failure mode table.
    fail_mode_columns = ["desc", "rpn", "frequency", "severity", "detection", "lower_bound",
                         "best_estimate", "upper_bound", "mission_time"]
    horizontal_header_labels = [
                "Failure Modes",
                "RPN",
                "Frequency",
                "Severity",
                "Detectability",
                "Lower Bound (LB)",
                "Best Estimate (BE)",
                "Upper Bound (UB)",
                "Mission Time"
            ]

    """

    Name: __init__
    Type: function
    Description: Initializes baseline PyQt5 framework for connecting pieces of the GUI together.

    """

    def __init__(self):
        # Initializes DataFrames with default values.
        self.read_sql_default()

        # function call to read in the database.csv file before running the rest of the gui
        self.read_data_from_csv()

        super().__init__()
        self.setWindowTitle("Component Failure Modes and Effects Analysis (FMEA)")
        self.setGeometry(100, 100, 1000, 572)
        self.setStyleSheet(
            "QPushButton { color: white; background-color: #C02F1D; }"
            "QLabel { color: #C02F1D; font-weight: bold; }"
            "QTableWidget { gridline-color: #C02F1D; } "
            "QLineEdit { border: 2px solid #C02F1D; }"
        )

        # Creating tabs widget
        self.central_widget = QTabWidget(self)
        self.setCentralWidget(self.central_widget)

        # Creating the tabs
        self.user_instructions_tab = QWidget()  # Create a new tab
        self.central_widget.addTab(
            self.user_instructions_tab, "User Instructions"
        )  # Add the tab to the QTabWidget
        self.main_tool_tab = QWidget()  # Create a new tab
        self.central_widget.addTab(
            self.main_tool_tab, "Main Tool"
        )  # Add the tab to the QTabWidget
        self.statistics_tab = QWidget()  # Create a new tab
        self.central_widget.addTab(
            self.statistics_tab, "Statistics"
        )  # Add the tab to the QTabWidget
        self.database_view_tab = QWidget()  # Create a new tab
        self.central_widget.addTab(
            self.database_view_tab, "Database View"
        )  # Add the tab to the QTabWidget

        self._init_instructions_tab()

        self._init_database_view_tab()

        self._init_main_tab()

        self._init_stats_tab()

        self.counter = 0
        self.questions = [
            "Does this system have redundancy, i.e. multiple units of the same component/subsystem in the case one fails?",
            "Does this system have diversity, i.e. multiple components/subsystems that are responsible for the same function?",
            "Does this system have safety features, e.g. sensors, user-warnings, fail-safes?",
        ]
        self.qindex = 0

    def _init_instructions_tab(self):
        ### START OF USER INSTRUCTIONS TAB SETUP ###

        # Create a QVBoxLayout instance
        layout = QVBoxLayout()

        # Create a QLabel instance and set the text
        instructions1 = QLabel("Welcome to the Component FMEA Risk Mitigation Tool!")
        instructions1.setAlignment(Qt.AlignCenter)  # Center the text
        instructions1.setStyleSheet(
            "QLabel {font-size: 30px;}"
        )  # Set the text color to black and increase the font size

        # Create QLabel instances for the logos
        logo1 = QLabel()
        logo2 = QLabel()

        # Create QPixmap instances with the logo images
        pixmap1 = QPixmap("images/Collins_Aerospace_Logo.png").scaled(
            100, 100, Qt.KeepAspectRatio
        )
        pixmap2 = QPixmap("images/WPI_Inst_Prim_FulClr.png").scaled(
            100, 100, Qt.KeepAspectRatio
        )

        # Set the QPixmap to the QLabel
        logo1.setPixmap(pixmap1)
        logo2.setPixmap(pixmap2)

        self.instructions2 = QWidget
        self.instructions2 = QTextEdit()
        self.instructions2.setText(
            """
        Please choose whether you want to complete an FMEA or FMECA analysis by clicking one of the buttons on the right!
        
        Here is some information regarding use:
        1. The “Main Tool” Tab allows the user to input a component name and a risk acceptance threshold and subsequently generate a table with that component’s failure modes and associated attributes in the database. These attributes include RPN, Frequency, Severity, Detection, and Lower Bound, Best Estimate, and Upper Bound (Failures Per Million Hours), all of which may-or-may-not have previously established values and are user-editable.
            - By filling in and saving the RPN values, the user is able to generate a bar and pie chart of those failure modes at any given time. The charts dynamically update with the table and can be regenerated via their respective buttons.
            - By filling in and saving the Frequency, Severity, and Detection values, the user is able to generate a 3D risk profile (a cuboid) and a 3D scatterplot of those failure modes at any given time. The color of these 3D plots ranges from green, yellow, and red to respectively reflect low, medium, and high-risk classifications extrapolated from a risk matrix.
            - The Lower Bound, Best Estimate, and Upper Bound values, given in Failures Per Million Hours (FPMH), are provided to guide the user to make an informed decision about an acceptable frequency value for its component. Furthermore, by hovering the mouse of the “Frequency” input text box, the user can receive a suggested frequency value based on the built-in “Humanoid in the Loop” mechanic.
    
        2. The “Database” tab allows the user to browse the entire database of components and their values for RPN, Frequency, Severity, Detection, Lower Bound, Best Estimate, and Upper Bound. Unestablished RPN values are automatically set to 1, while every other unestablished value is automatically set to 0. While component attribute values are editable, the user must always input a known component name from the database.

        3. The “Statistics tab allows the user to generate a neural network, regression, etc. [WORK IN PROGRESS]
        
        """
        )

        font = QFont()
        font.setPointSize(16)  # Set this to the desired size
        self.instructions2.setFont(font)

        # Set textEdit as ReadOnly
        self.instructions2.setReadOnly(True)

        # Create a QHBoxLayout for the header
        header_layout = QHBoxLayout()

        # Add the buttons to the header layout
        header_layout.addWidget(logo1)
        header_layout.addStretch(2)  # Add flexible space
        header_layout.addWidget(instructions1)
        header_layout.addStretch(2)  # Add flexible space
        header_layout.addWidget(logo2)

        # adding fmea/fmeca buttons to layout
        fmea_button = QPushButton("FMEA")
        fmeca_button = QPushButton("FMECA")
        logos_buttons_layout = QVBoxLayout()
        logos_buttons_layout.addWidget(fmea_button)
        logos_buttons_layout.addWidget(fmeca_button)

        # Add the header layout and the instructions2 widget to the main layout
        header_layout.addLayout(logos_buttons_layout)
        layout.addLayout(header_layout)
        layout.addWidget(self.instructions2)

        # Set the layout on the User Instructions tab
        self.user_instructions_tab.setLayout(layout)

        ### END OF USER INSTRUCTIONS TAB SETUP ###

    def _init_database_view_tab(self):
        ### START OF DATABASE VIEW TAB SETUP ###

        # Create the database view layout
        self.database_view_layout = QVBoxLayout(self.database_view_tab)

        # Create and add the read database button
        self.read_database_button = QPushButton("Read Database")
        self.read_database_button.clicked.connect(self.read_database)
        self.database_view_layout.addWidget(self.read_database_button)

        # Create and add the table widget for database view
        self.database_table_widget = QTableWidget()
        self.database_view_layout.addWidget(self.database_table_widget)

        ### END OF DATABASE VIEW TAB SETUP ###

    def _init_main_tab(self):
        ### START OF MAIN TAB SETUP ###

        # Create the main layout for the Main Tool tab
        self.main_layout = QHBoxLayout(self.main_tool_tab)

        # Create the left layout for input fields and table
        self.left_layout = QVBoxLayout()

        # Creating label to designate component selection
        self.component_selection = QLabel("Component Selection: ")
        self.left_layout.addWidget(self.component_selection)

        # Create and add the component name dropdown menu
        self.component_name_field = QComboBox(self)
        self.component_name_field.addItem("Select a Component")
        self.component_name_field.activated.connect(
            lambda: (
                self.component_name_field_stats.setCurrentText(
                    self.component_name_field.currentText()
                )
            )
        )
        for name in self.components["name"]:
            self.component_name_field.addItem(name)
        self.left_layout.addWidget(self.component_name_field)

        # Create and add the risk acceptance threshold field
        self.threshold_label = QLabel("Risk Acceptance Threshold:")
        self.threshold_field = QLineEdit()
        self.threshold_field.setText(str(DEFAULT_RISK_THRESHOLD))
        self.threshold_field.setToolTip(
            "Enter the maximum acceptable RPN: must be a value between [1-1000]."
        )
        self.left_layout.addWidget(self.threshold_label)
        self.left_layout.addWidget(self.threshold_field)

        # Create and add the submit button
        self.submit_button = QPushButton("Generate Table")
        self.submit_button.clicked.connect(self.show_table)
        self.left_layout.addWidget(self.submit_button)

        # Create and add the table widget
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(len(self.horizontal_header_labels))
        self.table_widget.setHorizontalHeaderLabels(self.horizontal_header_labels)
        self.table_widget.setColumnWidth(0, 150)  # ID
        self.table_widget.setColumnWidth(1, 150)  # Failure Mode
        self.table_widget.setColumnWidth(3, 150)  # RPN
        self.table_widget.setColumnWidth(4, 150)  # Frequency
        self.table_widget.setColumnWidth(5, 150)  # Severity
        self.table_widget.setColumnWidth(6, 150)  # Detectability
        self.table_widget.setColumnWidth(7, 150)  # Mission Time
        self.table_widget.setColumnWidth(8, 150)  # Lower Bound
        self.table_widget.setColumnWidth(9, 150)  # Best Estimate
        self.table_widget.setColumnWidth(10, 150)  # Upper Bound
        self.table_widget.verticalHeader().setDefaultSectionSize(32)
        self.table_widget.verticalHeader().setMaximumSectionSize(32)
        self.table_widget.verticalScrollBar().setMaximum(10 * 30)
        self.left_layout.addWidget(self.table_widget)

        # Add the left layout to the main layout
        self.main_layout.addLayout(self.left_layout)
        self.main_layout.setStretchFactor(
            self.left_layout, 2
        )  # make the left layout 3 times wider than the right

        # Create the right layout for the chart
        self.right_layout = QVBoxLayout()

        # Create label for the graphing
        self.plotting_label = QLabel("Data Visualization: ")
        self.right_layout.addWidget(self.plotting_label)

        # Create dropdown menu for holding charts we want to give the option of generating
        self.chart_name_field_main_tool = QComboBox(self)
        self.chart_name_field_main_tool.addItem("Select a Chart")
        self.chart_name_field_main_tool.addItem("Bar Chart")
        self.chart_name_field_main_tool.addItem("Pie Chart")
        self.chart_name_field_main_tool.addItem("3D Risk Plot")
        self.chart_name_field_main_tool.addItem("Scatterplot")
        self.chart_name_field_main_tool.addItem("Bubbleplot")
        self.right_layout.addWidget(self.chart_name_field_main_tool)

        # Create the matplotlib figure and canvas
        self.main_figure = plt.figure()
        self.canvas = FigureCanvas(self.main_figure)
        self.right_layout.addWidget(self.canvas)

        # Create and add the generate chart button
        self.generate_chart_button = QPushButton("Generate Chart")
        self.generate_chart_button.clicked.connect(self.generate_main_chart)
        self.right_layout.addWidget(self.generate_chart_button)

        # Create and add the download chart button
        self.download_chart_button = QPushButton("Download Chart")
        self.download_chart_button.clicked.connect(
            lambda: self.download_chart(self.main_figure)
        )
        self.right_layout.addWidget(self.download_chart_button)

        # Add a stretch to the right layout
        self.right_layout.addStretch()

        # Add the right layout to the main layout
        self.main_layout.addLayout(self.right_layout)
        self.main_layout.setStretchFactor(
            self.right_layout, 2
        )  # the right layout stays with default size

        # Create a QHBoxLayout for the navigation buttons
        self.nav_button_layout = QHBoxLayout()

        # Initialize the selected index and the maximum number of IDs to show
        self.selected_index = 0
        self.max_ids = 10

        # Create and connect the navigation buttons
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous)
        self.nav_button_layout.addWidget(self.prev_button)

        # Add spacing between the buttons
        self.nav_button_layout.addSpacing(10)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next)
        self.nav_button_layout.addWidget(self.next_button)

        # Add the nav_button_layout to the left layout
        self.left_layout.addLayout(self.nav_button_layout)

        # Create and add the save button
        self.save_button = QPushButton("Save RPN Values")
        self.save_button.clicked.connect(self.save_values)
        self.left_layout.addWidget(self.save_button)

        # Create and add the X, Y, and Z input fields and 3D plot
        # self.input_and_plot_layout = QVBoxLayout()
        self.x_input_label = QLabel("Probability :")
        self.x_input_field = QLineEdit()
        self.x_input_field.setValidator(
            QDoubleValidator(0.99, 99.99, 2)
        )  # This will only accept float values
        self.x_input_field.setToolTip("Humanoid Recommendation: 9 or 10 (Unacceptable)")
        self.left_layout.addWidget(self.x_input_label)
        self.left_layout.addWidget(self.x_input_field)

        self.y_input_label = QLabel("Severity:")
        self.y_input_field = QLineEdit()
        self.y_input_field.setValidator(
            QDoubleValidator(0.99, 99.99, 2)
        )  # This will only accept float values
        self.left_layout.addWidget(self.y_input_label)
        self.left_layout.addWidget(self.y_input_field)

        self.z_input_label = QLabel("Detection:")
        self.z_input_field = QLineEdit()
        self.z_input_field.setValidator(
            QDoubleValidator(0.99, 99.99, 2)
        )  # This will only accept float values
        self.left_layout.addWidget(self.z_input_label)
        self.left_layout.addWidget(self.z_input_field)

        self.table_widget.cellClicked.connect(self.cell_clicked)

        # Connect QLineEdit's textChanged signal to the on_rpn_item_changed
        self.x_input_field.textChanged.connect(self.on_rpn_item_changed)
        self.y_input_field.textChanged.connect(self.on_rpn_item_changed)
        self.z_input_field.textChanged.connect(self.on_rpn_item_changed)

        ### END OF MAIN TAB SETUP ###

    def _init_stats_tab(self):
        ### START OF STATISTICS TAB SETUP ###

        # Create main layout
        main_layout_stats = QHBoxLayout(self.statistics_tab)

        # Left layout for component info and table
        left_layout_stats = QVBoxLayout()

        # Creating label to designate component selection
        self.component_selection_stats = QLabel("Component Selection: ")
        left_layout_stats.addWidget(self.component_selection_stats)

        # Create and add the component name dropdown menu
        self.component_name_field_stats = QComboBox(self)
        self.component_name_field_stats.addItem("Select a Component")
        self.component_name_field_stats.activated.connect(
            lambda: (
                self.component_name_field.setCurrentText(
                    self.component_name_field_stats.currentText()
                )
            )
        )
        for name in self.components["name"]:
            self.component_name_field_stats.addItem(name)
        self.left_layout.addWidget(self.component_name_field_stats)
        left_layout_stats.addWidget(self.component_name_field_stats)

        # Create and add the submit button
        self.stat_submit_button = QPushButton("Show Table")
        self.stat_submit_button.clicked.connect(self.show_table_stats)
        left_layout_stats.addWidget(self.stat_submit_button)

        # Create button for detectability recommendation
        self.detectability_button_stats = QPushButton(
            "Get Detectability Recommendation"
        )
        self.detectability_button_stats.clicked.connect(self.ask_questions)
        left_layout_stats.addWidget(self.detectability_button_stats)

        # Create and add the table widget
        self.table_widget_stats = QTableWidget()
        self.table_widget_stats.setColumnCount(len(self.horizontal_header_labels))
        self.table_widget_stats.setHorizontalHeaderLabels(self.horizontal_header_labels)
        self.table_widget_stats.setColumnWidth(0, 150)  # ID
        self.table_widget_stats.setColumnWidth(1, 150)  # Failure Mode
        self.table_widget_stats.setColumnWidth(3, 150)  # RPN
        self.table_widget_stats.setColumnWidth(4, 150)  # Frequency
        self.table_widget_stats.setColumnWidth(5, 150)  # Severity
        self.table_widget_stats.setColumnWidth(6, 150)  # Detectability
        self.table_widget_stats.setColumnWidth(7, 150)  # Mission Time
        self.table_widget_stats.setColumnWidth(8, 150)  # Lower Bound
        self.table_widget_stats.setColumnWidth(9, 150)  # Lower Bound
        self.table_widget_stats.setColumnWidth(10, 150)  # Best Estimate
        self.table_widget_stats.verticalHeader().setDefaultSectionSize(32)
        self.table_widget_stats.verticalHeader().setMaximumSectionSize(32)
        self.table_widget_stats.verticalScrollBar().setMaximum(10 * 30)
        left_layout_stats.addWidget(self.table_widget_stats)

        # Create a QHBoxLayout for the navigation buttons
        self.nav_button_layout_stats = QHBoxLayout()

        # Initialize the selected index and the maximum number of IDs to show
        self.selected_index_stats = 0
        self.max_ids_stats = 10

        # Create and connect the navigation buttons
        self.prev_button_stats = QPushButton("Previous")
        self.prev_button_stats.clicked.connect(self.show_previous_stats)
        self.nav_button_layout_stats.addWidget(self.prev_button_stats)

        # Add spacing between the buttons
        self.nav_button_layout_stats.addSpacing(10)

        self.next_button_stats = QPushButton("Next")
        self.next_button_stats.clicked.connect(self.show_next_stats)
        self.nav_button_layout_stats.addWidget(self.next_button_stats)

        # Add the nav_button_layout to the left layout
        left_layout_stats.addLayout(self.nav_button_layout_stats)

        # Creating right layout for graphs in stats tab
        right_layout_stats = QVBoxLayout()

        # Create label for the graphing
        self.stat_modeling_tag = QLabel("Statistical Modeling: ")
        right_layout_stats.addWidget(self.stat_modeling_tag)

        # Create dropdown menu for holding charts we want to give the option of generating
        self.chart_name_field_stats = QComboBox(self)
        self.chart_name_field_stats.addItem("Select a Chart")
        self.chart_name_field_stats.addItem("Weibull Distribution")
        self.chart_name_field_stats.addItem("Rayleigh Distribution")
        self.chart_name_field_stats.addItem("Bathtub Curve")
        right_layout_stats.addWidget(self.chart_name_field_stats)

        # Matplotlib canvases with tab widget (hardcoded for one component)
        self.stats_tab = QTabWidget()
        self.stats_tab_canvas1 = FigureCanvas(Figure())
        self.stats_tab_canvas2 = FigureCanvas(Figure())
        self.stats_tab_canvas3 = FigureCanvas(Figure())
        self.stats_tab.addTab(self.stats_tab_canvas1, "Failure Mode 1")
        self.stats_tab.addTab(self.stats_tab_canvas2, "Failure Mode 2")
        self.stats_tab.addTab(self.stats_tab_canvas3, "Failure Mode 3")
        right_layout_stats.addWidget(self.stats_tab)

        # Create and add the generate chart button
        self.generate_chart_button_stats = QPushButton("Generate Chart")
        self.generate_chart_button_stats.clicked.connect(self.generate_stats_chart)
        right_layout_stats.addWidget(self.generate_chart_button_stats)

        # Create and add the download chart button (non-functional)
        self.download_chart_button_stats = QPushButton("Download Chart")
        self.download_chart_button_stats.clicked.connect(
            lambda: self.download_chart(self.stats_tab_canvas1.figure)
        )
        right_layout_stats.addWidget(self.download_chart_button_stats)

        # Add left and right layouts to the main layout
        main_layout_stats.addLayout(left_layout_stats, 4)
        main_layout_stats.addLayout(right_layout_stats, 6)

        ### END OF STATISTICS TAB SETUP ###

    """

    Name: generate_chart
    Type: function
    Description: Uses dropdown menu to generate the desired chart in the main tool tab.

    """

    def generate_main_chart(self):
        match (self.chart_name_field_main_tool.currentText()):
            case "Bar Chart":
                self.bar_chart()
            case "Pie Chart":
                self.pie_chart()
            case "3D Risk Plot":
                self.plot_3D()
            case "Scatterplot":
                self.scatterplot()
            case "Bubbleplot":
                self.bubble_plot()

    def generate_stats_chart(self):
        match (self.chart_name_field_stats.currentText()):
            case "Weibull Distribution":
                self.update_weibull_canvas()
            case "Rayleigh Distribution":
                self.update_rayleigh_canvas()
            case "Bathtub Curve":
                self.update_bathtub_canvas()

    """

    Name: bathtub
    Type: function
    Description: Invokes the bathtub function in stats.py to populate the canvas with a histogram PDF plot.

    """

    def update_bathtub_canvas(self):
        N = 1000
        T = 20
        t1 = 1
        t2 = 10

        self.stats_tab_canvas1.figure.clear()
        self.stats_tab_canvas2.figure.clear()
        self.stats_tab_canvas3.figure.clear()

        self.stats_tab.clear()

        fig1 = stats._bathtub(N, T, t1, t2)
        fig2 = stats._bathtub(N, T, t1, t2)
        fig3 = stats._bathtub(N, T, t1, t2)

        self.stats_tab_canvas1.figure = fig1
        self.stats_tab_canvas1.figure.tight_layout()
        self.stats_tab_canvas1.draw()
        self.stats_tab_canvas2.figure = fig2
        self.stats_tab_canvas2.figure.tight_layout()
        self.stats_tab_canvas2.draw()
        self.stats_tab_canvas3.figure = fig3
        self.stats_tab_canvas3.figure.tight_layout()
        self.stats_tab_canvas3.draw()

        # Add tabs after generating the graphs
        self.stats_tab.addTab(self.stats_tab_canvas1, "Plot 1")
        self.stats_tab.addTab(self.stats_tab_canvas2, "Plot 2")
        self.stats_tab.addTab(self.stats_tab_canvas3, "Plot 3")

    """
    
    Name: update_rayleigh_canvas
    Type: function
    Description: Invokes the rayleigh function in stats.py to populate the canvas with a histogram PDF plot.
    
    """

    def update_rayleigh_canvas(self):
        # Clear the existing figures before displaying new ones
        self.stats_tab_canvas1.figure.clear()
        self.stats_tab_canvas2.figure.clear()
        self.stats_tab_canvas3.figure.clear()

        # Clear the existing tabs
        self.stats_tab.clear()

        fig1 = stats._rayleigh(self.values())
        fig2 = stats._rayleigh(self.values())
        fig3 = stats._rayleigh(self.values())
        """
        if (
            self.component_name_field.currentText() == "Motor-Driven Pump"
        ):  # Idealy, this would be modified to search through the list of component neames instead of the names being specificed here
            # Get new figures from weibull()
            fig1 = stats._rayleigh(np.array([20.8, 125.0, 4.17]))
            print(fig1)
            fig2 = stats._rayleigh(np.array([1.0, 30.0, 1000.0]))
            fig3 = stats._rayleigh(np.array([4.17, 83.3, 417.0]))
        if (
            self.component_name_field.currentText() == "Motor-Operated Valves"
        ):  # Same here
            fig1 = stats._rayleigh(np.array([41.7, 125.0, 375.0]))
            fig2 = stats._rayleigh(np.array([83.3, 4170.0, 4.17]))
            fig3 = stats._rayleigh(np.array([2.5, 33.3, 250.0]))
        """

        # Update the canvas with the new figures
        self.stats_tab_canvas1.figure = fig1
        self.stats_tab_canvas1.figure.tight_layout()
        self.stats_tab_canvas1.draw()
        self.stats_tab_canvas2.figure = fig2
        self.stats_tab_canvas2.figure.tight_layout()
        self.stats_tab_canvas2.draw()
        self.stats_tab_canvas3.figure = fig3
        self.stats_tab_canvas3.figure.tight_layout()
        self.stats_tab_canvas3.draw()

        # Add tabs after generating the graphs
        self.stats_tab.addTab(self.stats_tab_canvas1, "Plot 1")
        self.stats_tab.addTab(self.stats_tab_canvas2, "Plot 2")
        self.stats_tab.addTab(self.stats_tab_canvas3, "Plot 3")

    """
    
    Name: updateWeibullCavas
    Type: function
    Description: Invokes the weibull function in stats.py to populate the cavas with a histogram PDF plot.
    
    """

    def update_weibull_canvas(self):
        # Clear the existing figures before displaying new ones
        self.stats_tab_canvas1.figure.clear()
        self.stats_tab_canvas2.figure.clear()
        self.stats_tab_canvas3.figure.clear()

        # Clear the existing tabs
        self.stats_tab.clear()

        fig1 = stats._weibull(self.values())
        fig2 = stats._weibull(self.values())
        fig3 = stats._weibull(self.values())

        """
        if (
            self.component_name_field_stats.currentText() == "Motor-Driven Pump"
        ):  # Fix to read through database
            # Get new figures from weibull()
            fig1 = stats._weibull(np.array([20.8, 125.0, 4.17]))
            fig2 = stats._weibull(np.array([1.0, 30.0, 1000.0]))
            fig3 = stats._weibull(np.array([4.17, 83.3, 417.0]))
        if (
            self.component_name_field_stats.currentText() == "Motor-Operated Valves"
        ):  # Fix to read through database (likely with currently unused values() function in stats.py)
            fig1 = stats._weibull(np.array([41.7, 125.0, 375.0]))
            fig2 = stats._weibull(np.array([83.3, 4170.0, 4.17]))
            fig3 = stats._weibull(np.array([2.5, 33.3, 250.0]))
        """

        # Update the canvas with the new figures
        self.stats_tab_canvas1.figure = fig1
        self.stats_tab_canvas1.figure.tight_layout()
        self.stats_tab_canvas1.draw()
        self.stats_tab_canvas2.figure = fig2
        self.stats_tab_canvas2.figure.tight_layout()
        self.stats_tab_canvas2.draw()
        self.stats_tab_canvas3.figure = fig3
        self.stats_tab_canvas3.figure.tight_layout()
        self.stats_tab_canvas3.draw()

        self.stats_tab.addTab(self.stats_tab_canvas1, "Plot 1")
        self.stats_tab.addTab(self.stats_tab_canvas2, "Plot 2")
        self.stats_tab.addTab(self.stats_tab_canvas3, "Plot 3")

    """

    Name: read_database
    Type: function
    Description: Function that initially clears the table and then repopulates it.

    """

    def read_database(self):
        self.database_table_widget.clear()

        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, "database.csv")

            with open(file_path, newline="") as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)

                # Set the row count and column count for the table widget
                self.database_table_widget.setRowCount(len(rows))
                self.database_table_widget.setColumnCount(len(rows[0]))

                # Set the table headers
                headers = rows[0]
                self.database_table_widget.setHorizontalHeaderLabels(headers)

                # Populate the table widget with data
                for row_idx, row in enumerate(rows[1:]):
                    for col_idx, item in enumerate(row):
                        table_item = QTableWidgetItem(item)
                        self.database_table_widget.setItem(row_idx, col_idx, table_item)
        except FileNotFoundError:
            error_message = "Error: Could not find the database.csv file."
            QMessageBox.critical(self, "File Not Found", error_message)

    """

    Name: read_data_from_csv
    Type: function
    Description: Clears local database and repopulates it.

    """

    def read_data_from_csv(self):
        database_data.clear()
        self.database_data = {}
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "database.csv")

        # error checking for the existence of a database.csv file
        try:
            with open(file_path, newline="") as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # skip the header row
                for row in reader:
                    # setting component name to be the element at position [X , ...]
                    component_name = row[0]
                    # creating the element in the database for a new component if not already present
                    if component_name not in database_data:
                        database_data[component_name] = []
                    # filling the list for holding information for a component's failure modes
                    database_data[component_name].append(
                        {
                            "id": int(row[1]),
                            "failure_mode": row[2],
                            "rpn": float(row[3]),
                            "lower_bound": float(row[4]),
                            "best_estimate": float(row[5]),
                            "upper_bound": float(row[6]),
                            "frequency": float(row[7]),
                            "severity": float(row[8]),
                            "detectability": float(row[9]),
                            "mission_time": float(row[10]),
                        }
                    )
        except FileNotFoundError:
            error_message = "Error: Could not find the database.csv file."
            QMessageBox.critical(self, "File Not Found", error_message)

        return database_data

    """
    Pulls default data from part_info.db and stores it in a pandas DataFrame.
    """

    def read_sql_default(self) -> None:
        default_db_path = os.path.abspath(
            os.path.join(self.current_directory, self.db_path, self.default_db_name)
        )
        if not os.path.isfile(default_db_path):
            error_message = "Error: could not find part_info.db."
            QMessageBox.critical(self, "File Not Found", error_message)
            return
        self.default_conn = sqlite3.connect(default_db_path)
        self.components = pd.read_sql_query(
            "SELECT * FROM components", self.default_conn
        )
        self.fail_modes = pd.read_sql_query(
            "SELECT * FROM fail_modes", self.default_conn
        )
        self.comp_fails = pd.read_sql_query(
            "SELECT * FROM comp_fails", self.default_conn
        )
        # Calculates RPN = Frequency * Severity * Detection
        self.comp_fails.insert(
            2,
            "rpn",
            [
                int(row["frequency"] * row["severity"] * row["detection"])
                for _, row in self.comp_fails.iterrows()
            ],
            True,
        )

    def read_risk_threshold(self):
        try:
            risk_threshold = float(self.threshold_field.text())
            # error checking for risk value threshold
            if not (1 <= risk_threshold <= 1000):
                error_message = "Error: Please re-enter a risk threshold value between 1 and 1000, inclusive."
                QMessageBox.critical(self, "Value Error", error_message)
        except:
            risk_threshold = DEFAULT_RISK_THRESHOLD
        return risk_threshold

    """

    Name: show_table
    Type: function
    Description: Clears table widget and fills table in with the desired data for the input component.

    """

    def show_table(self):
        self.populate_table(self.table_widget)

    """
    Populates a table with failure modes associated with a specific component.
    """

    def populate_table(self, table_widget) -> None:
        # clear existing table data (does not affect underlying database)
        table_widget.clearContents()

        # retrieve component name from text box
        component_name = self.component_name_field.currentText()

        # Error checking for component name
        if not self.components["name"].str.contains(component_name).any():
            error_message = "Error: Please re-enter a component name that's present in the database."
            QMessageBox.critical(self, "Name Error", error_message)

        # Update the column header for "Failure Mode"
        table_widget.setHorizontalHeaderLabels(
            [f"{component_name} Failure Modes"] + self.horizontal_header_labels[1:]
        )

        # Update the maximum number of IDs to show
        self.max_ids = 10

        # drop_duplicates shouldn't be necessary here, since components are unique. Just in case, though.
        # np.sum is a duct-tapey way to convert to int, since you can't directly
        comp_id = np.sum(self.components[self.components["name"] == component_name].drop_duplicates()["id"])
        component_data2 = self.comp_fails[self.comp_fails["comp_id"] == comp_id].head(self.max_ids).reset_index(drop=True)
        component_data2 = pd.merge(self.fail_modes, component_data2, left_on="id", right_on="fail_id")

        # Get risk acceptance threshold
        risk_threshold = self.read_risk_threshold()

        # Set the row count of the table widget
        table_widget.setRowCount(self.max_ids)

        for row, data in component_data2.iterrows():
            for i, key in enumerate(self.fail_mode_columns):
                table_widget.setItem(row, i, QTableWidgetItem(str(data[key])))


    """
    Clears table widget in stats tab and fills table in with the desired data for the input component.
    """

    def show_table_stats(self) -> None:
        self.populate_table(self.table_widget_stats)
       
    """
    Records the location of a cell when it's clicked.
    """

    def cell_clicked(self, row, column):
        self.current_row = row
        self.current_column = column

    """
    TODO: get lower bound, geometric mean, and upper bound from dataset, for the component passed in
    """

    def values(self):
        print("generating values")
        component_name = self.component_name_field.currentText()

        values1 = np.array(
            [20.8, 125.0, 4.17]
        )  # lower bound, geometric mean, and upper bound
        values2 = np.array([1.0, 30.0, 1000.0])
        values3 = np.array([4.17, 83.3, 417.0])
        values4 = np.array([41.7, 125.0, 375.0])
        values5 = np.array([83.3, 4170.0, 4.17])
        values6 = np.array([2.5, 33.3, 250.0])

        if component_name == "Motor-Driven Pump":
            return values1
        elif component_name == "Motor-Operated Valves":
            return values4
        else:
            return np.array([1, 1, 1])

    """
    Retrieves frequency, severity, and detection values and calculates new RPN which is 
    pushed to the table widget.
    """

    def on_rpn_item_changed(self):
        # Get the Frequency, Severity, and Detection values
        probability = (
            float(self.x_input_field.text()) if self.x_input_field.text() else 0
        )
        severity = float(self.y_input_field.text()) if self.y_input_field.text() else 0
        detection = float(self.z_input_field.text()) if self.z_input_field.text() else 0

        # Get the risk acceptance threshold
        risk_threshold = self.read_risk_threshold()

        # Calculate the RPN
        rpn = probability * severity * detection

        # error checking for RPN value
        if (probability or severity or detection) < 1 or (
            probability or severity or detection
        ) > 10:
            error_message = "Error: Please re-enter values between 1 and 10, inclusive."
            QMessageBox.critical(self, "Value Error", error_message)
            return

        # Update the RPN value in the table/chart
        rpn_item = self.table_widget.item(self.current_row, 2)
        frequency_item = self.table_widget.item(self.current_row, 3)
        severity_item = self.table_widget.item(self.current_row, 4)
        detectability_item = self.table_widget.item(self.current_row, 5)

        if rpn_item:
            rpn_item.setText(str(rpn))
            if rpn > risk_threshold:
                rpn_item.setBackground(QColor(255, 102, 102))  # muted red
            else:
                rpn_item.setBackground(QColor(102, 255, 102))  # muted green

        if frequency_item:
            frequency_item.setText(str(probability))
        if severity_item:
            severity_item.setText(str(severity))
        if detectability_item:
            detectability_item.setText(str(detection))

    """
    Saves RPN, frequency, severity, and detectability values to the local database.
    """

    def save_values(self):
        component_name = self.component_name_field.currentText()
        component_data = database_data.get(component_name, [])

        for row in range(self.table_widget.rowCount()):
            rpn_item = self.table_widget.item(row, 2)
            frequency_item = self.table_widget.item(row, 3)
            severity_item = self.table_widget.item(row, 4)
            detectability_item = self.table_widget.item(row, 5)
            mission_time = self.table_widget.item(row, 6)

            if rpn_item:
                component_data[row + self.selected_index]["rpn"] = (
                    float(frequency_item.text())
                    * float(mission_time.text())
                    * float(severity_item.text())
                    * float(detectability_item.text())
                )
            if frequency_item:
                component_data[row + self.selected_index]["frequency"] = float(
                    frequency_item.text()
                )
            if severity_item:
                component_data[row + self.selected_index]["severity"] = float(
                    severity_item.text()
                )
            if detectability_item:
                component_data[row + self.selected_index]["detectability"] = float(
                    detectability_item.text()
                )
            if mission_time:
                component_data[row + self.selected_index]["mission_time"] = float(
                    mission_time.text()
                )

        # Update the database with the modified component data
        database_data[component_name] = component_data

    """
    Refreshes table to the previous page.
    """

    def show_previous(self):
        # so that it doesn't go below 0
        self.selected_index = max(0, self.selected_index - self.max_ids)
        self.show_table()

    """
    Description: Refreshes statistics table to the previous page.
    """

    def show_previous_stats(self):
        # so that it doesn't go below 0
        self.selected_index_stats = max(
            0, self.selected_index_stats - self.max_ids_stats
        )
        self.show_table_stats()

    """
    Refreshes table to the next page.
    """

    def show_next(self):
        component_name = self.component_name_field.currentText()
        component_data = database_data.get(component_name, [])
        total_pages = len(component_data) // self.max_ids
        if len(component_data) % self.max_ids != 0:
            total_pages += 1

        if self.selected_index + self.max_ids < len(component_data):
            self.selected_index += self.max_ids
        elif (
            self.selected_index + self.max_ids >= len(component_data)
            and self.selected_index // self.max_ids < total_pages - 1
        ):
            self.selected_index = (total_pages - 1) * self.max_ids
        self.show_table()

    """
    Refreshes statistics table to the next page.
    """

    def show_next_stats(self):
        component_name = self.component_name_field_stats.currentText()
        component_data = database_data.get(component_name, [])
        total_pages = len(component_data) // self.max_ids_stats
        if len(component_data) % self.max_ids_stats != 0:
            total_pages += 1

        if self.selected_index_stats + self.max_ids_stats < len(component_data):
            self.selected_index_stats += self.max_ids_stats
        elif (
            self.selected_index_stats + self.max_ids_stats >= len(component_data)
            and self.selected_index_stats // self.max_ids_stats < total_pages - 1
        ):
            self.selected_index_stats = (total_pages - 1) * self.max_ids_stats
        self.show_table_stats()

    """
    Refreshes displayed chart with new changes to the table.
    """

    def bar_chart(self):
        component_data = []
        threshold = float(self.threshold_field.text())

        for row in range(self.table_widget.rowCount()):
            id_item = self.table_widget.item(row, 0)
            failure_mode_item = self.table_widget.item(row, 1)
            rpn_item = self.table_widget.item(row, 2)
            if id_item and failure_mode_item and rpn_item:
                component_data.append(
                    {
                        "id": int(id_item.text()),
                        "failure_mode": failure_mode_item.text(),
                        "rpn": float(rpn_item.text()),
                    }
                )

        # Clear the existing plot
        self.main_figure.clear()

        # Adjust the subplot for spacing
        self.main_figure.subplots_adjust(
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
        ax = self.main_figure.add_subplot(111)

        # Set the color of the bars based on RPN values
        colors = ["#5f9ea0" if rpn < threshold else "#FF6961" for rpn in rpn_values]
        sns.barplot(x="Failure Mode ID", y="RPN", data=df, palette=colors, ax=ax)

        ax.axhline(threshold, color="#68855C", linestyle="--")
        ax.set_ylabel("Risk Priority Number (RPN)")
        ax.set_xlabel("Failure Mode ID")
        component_name = self.component_name_field.currentText()
        ax.set_title(component_name + " Risk Profile")
        ax.tick_params(axis="x", rotation=0)

        # Set the font to bold
        font = {"weight": "bold"}
        mpl.rc("font", **font)

        # Set the x-axis ticks to integers only
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Refresh the canvas
        self.canvas.draw()

    """
    Gives user the option to download displayed figure.
    """

    def download_chart(self, figure):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "JPEG (*.jpg);;All Files (*)"
        )
        self.main_figure
        figure.savefig(file_path, format="jpg", dpi=300)

    """
    Makes a pie chart of data in table.
    """

    def pie_chart(self):
        # Clear the existing plot
        self.main_figure.clear()

        component_data = []
        threshold = float(self.threshold_field.text())
        below_threshold = 0
        above_threshold = 0

        for row in range(self.table_widget.rowCount()):
            id_item = self.table_widget.item(row, 0)
            failure_mode_item = self.table_widget.item(row, 1)
            rpn_item = self.table_widget.item(row, 2)
            if id_item and failure_mode_item and rpn_item:
                rpn = float(rpn_item.text())
                if rpn < threshold:
                    below_threshold += 1
                else:
                    above_threshold += 1

        # Clear the existing plot
        self.main_figure.clear()

        # Prepare the data for the pie chart
        labels = ["Below Risk Threshold", "Above Risk Threshold"]
        rpn_values = [below_threshold, above_threshold]

        # Set the color of the slices based on the categories
        colors = ["#5f9ea0", "#FF6961"]

        # Create a pie chart
        ax = self.main_figure.add_subplot(111)
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

        component_name = self.component_name_field.currentText()
        ax.set_title(component_name + " Risk Profile")

        # Refresh the canvas
        self.canvas.draw()

    """
    Displays 3D plot of data in table.
    """

    def plot_3D(self):
        # Clear the existing plot
        self.main_figure.clear()

        # Get the X, Y, and Z values
        try:
            length = float(self.x_input_field.text())
            width = float(self.y_input_field.text())
            height = float(self.z_input_field.text())
        except ValueError:
            QMessageBox.critical(
                self,
                "Value Error",
                "Please enter valid numbers for Frequency, Severity, and Detection.",
            )
            return

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
        self.main_figure.clear()
        ax = self.main_figure.add_subplot(111, projection="3d")

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

        self.canvas.draw()

    """
    Displays a 3D scatterplot of data (Frequency, Severity, Detection) in table.
    """

    def scatterplot(self):
        component_data = []
        threshold = float(self.threshold_field.text())

        for row in range(self.table_widget.rowCount()):
            id_item = self.table_widget.item(row, 0)
            frequency_item = self.table_widget.item(row, 3)
            severity_item = self.table_widget.item(row, 4)
            detection_item = self.table_widget.item(row, 5)
            if id_item and severity_item and detection_item and frequency_item:
                component_data.append(
                    {
                        "id": int(id_item.text()),
                        "severity": float(severity_item.text()),
                        "detection": float(detection_item.text()),
                        "frequency": float(frequency_item.text()),
                    }
                )

        # Clear the existing plot
        self.main_figure.clear()

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
        ax = self.main_figure.add_subplot(111, projection="3d")

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
        component_name = self.component_name_field.currentText()
        ax.set_title(component_name + " Risk Profile")

        # Add a colorbar
        self.main_figure.colorbar(sc, ax=ax, pad=0.02)

        # Refresh the canvas
        self.canvas.draw()

    """
    Makes a bubble chart of data in table. Builds upon the scatterplot function by altering bubbles to size according to RPN
    """

    def bubble_plot(self):
        component_data = []
        threshold = float(self.threshold_field.text())

        for row in range(self.table_widget.rowCount()):
            id_item = self.table_widget.item(row, 0)
            frequency_item = self.table_widget.item(row, 3)
            severity_item = self.table_widget.item(row, 4)
            detection_item = self.table_widget.item(row, 5)
            if id_item and severity_item and detection_item and frequency_item:
                component_data.append(
                    {
                        "id": int(id_item.text()),
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
        self.main_figure.clear()
        ax = self.main_figure.add_subplot(111, projection="3d")

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
        component_name = self.component_name_field.currentText()
        ax.set_title(component_name + " 3D Bubble Plot")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Severity")
        ax.set_zlabel("Detection")

        # Color bar which maps values to colors.
        cbar = self.main_figure.colorbar(bubble, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label("Risk Priority Number (RPN)")

        # plt.show()
        self.canvas.draw()

    def ask_questions(self):
        if self.qindex < len(self.questions):
            reply = QMessageBox.question(
                self,
                "Question",
                self.questions[self.qindex],
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if reply == QMessageBox.Yes:
                self.counter += 1

            self.qindex += 1
            self.ask_questions()
        else:
            self.show_recommendation()

    def show_recommendation(self):
        QMessageBox.information(
            self, "Recommendation", self.recommendations[self.counter]
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
