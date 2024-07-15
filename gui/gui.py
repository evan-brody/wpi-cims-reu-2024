"""

Project: AI-Based FMECA Tool
2023 Contributors: Nicholas Grabill, Stephanie Wang
2024 Contributors: Evan Brody, Karl Ramus, Esther Yu
Description: Implements the tool's GUI
Industry Mentor: Dr. Yehia Khalil
2023 Academic Mentors: Dr. Frank Zou, Dr. Tharindu De Alwis, Hammed Olayinka
2024 Academic Mentors: Dr. Frank Zou, Dr. Tharindu De Alwis, Yanping Pei, Grace Cao

File(s):
1) gui.py
    1a) Runs GUI implemented in Python w/ PyQt5 framework
2) statistics.py
    2a) Statistical modeling and analysis w/ framework described in paper
3) charts.py
    3a) Contains charts that can be generated 
4) part_info.db
    4a) Holds data for the GUI

NSF REU Project under grant DMS-2244306 in Industrial Mathematics & Statistics sponsored by Collins Aerospace and
Worcester Polytechnic Institute.

### TASK LIST ###

TODO: Add bathtub curve in Statistics tab with options for parameter changes. (WIP, just need to write some code)

TODO: UI Bug fixes
    TODO: Tables are same size so labels are cut off
    TODO: Variable plot sizes in stats tab
    TODO: detectability recommendation should reset when selected component is changed
    TODO: auto refresh on statistics page
    TODO: fix dependency arrow snapping when dragging rectangles over each other

TODO:
    # Success criteria: fails if n or more components fail
    # Eraser cursor
"""

import os, sys, sqlite3, logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stats_and_charts import stats
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import torch.multiprocessing as mp
from stats_and_charts.charts import Charts
import lstm.train_lstm as train_lstm
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from graph.dep_graph import DepGraph
from nlp import csv_loader_tab
from nlp import subtab
from nlp import similar

NUM_PROCESSES = 4

# There should be only one instance of this class
# It's the toolbar on the right side of the Dependency Analysis tab
class DepQToolBar(QToolBar):
    def __init__(self) -> None:
        super().__init__()

        self.selected_tool = None

# Instances of this class are the buttons on the
# right side of the Dependency Analysis tab
class DepQAction(QAction):
    def __init__(self, icon: QIcon, text: str, 
                 parent_toolbar: DepQToolBar,
                 parent_scene: QGraphicsScene, 
                 parent_window: QMainWindow) -> None:
        super().__init__(icon, text)

        self.parent_toolbar = parent_toolbar
        self.parent_scene = parent_scene
        self.parent_window = parent_window
        self.setCheckable(True)
        self.triggered.connect(self.clear_other_selections)

        self.parent_toolbar.addAction(self)

    # Checked is the new state
    def clear_other_selections(self, checked: bool) -> None:
        if not checked:
            return

        for action in self.parent_toolbar.actions():
            action.setChecked(action == self)
        self.parent_scene.dep_origin = None
        self.parent_scene.del_dyn_arr()

        self.parent_toolbar.selected_tool = self

        if self == self.parent_window.eraser_button:
            QApplication.setOverrideCursor(self.parent_window.eraser_cursor)
        else:
            QApplication.restoreOverrideCursor()

class DepQMenu(QMenu):
    def __init__(self, dg: DepGraph, parent_rect: QGraphicsRectItem, pos: QPoint) -> None:
        super().__init__()

        self.dg = dg
        self.parent_rect = parent_rect

        # Removes icons
        self.setStyleSheet(
            "QMenu::item {"
                "background-color: rgb(255, 255, 255);"
                "padding: 2px 5px 2px 2px;"
            "}"
            "QMenu::item:selected {"
                "background-color: rgb(0, 85, 127);"
                "color: rgb(255, 255, 255);"
            "}"
        )

        self.dr_action = self.addAction(f"Direct Risk: {self.dg.get_vertex_weight(self.parent_rect):.3f}")
        # NLP integration ?

        self.exec(pos)

class DepQComboBox(QComboBox):
    def __init__(self, parent_rect: QGraphicsRectItem, 
                 parent_scene: QGraphicsScene, 
                 parent_window: QMainWindow) -> None:
        super().__init__()

        self.parent_rect = parent_rect
        self.parent_scene = parent_scene
        self.parent_window = parent_window

        self.setEditable(True)
        self.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.setMinimumContentsLength(25)
        self.addItems(self.parent_window.components["name"])
        self.textActivated.connect(self.update_comp_fail_rate)

    def update_comp_fail_rate(self, comp_str: str) -> None:
        # Predict the failure rate using the RNN
        new_weight = train_lstm.predict(comp_str).item()
        self.parent_scene.dg.update_vertex(self.parent_rect, new_weight)

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()

# Custom QGraphicsScene class for the dependency tab
class DepQGraphicsScene(QGraphicsScene):
    # Keys for the QGraphicsItem data table
    MOUSE_DELTA = 0
    IS_COMPONENT = 1
    IS_AND_GATE = 2
    EDGES_VERTICES = 3
    RISK_LABEL = 4

    # The tip of a dependency arrow is an isosceles triangle
    ARR_LONG = 30  # The length of the middle axis
    ARR_SHORT = 15  # Half the length of the base

    RECT_DIMS = (400, 200)

    ERASER_RADIUS = 50

    SCENE_WIDTH = 5_000
    SCENE_HEIGHT = 1_000

    def __init__(self, parent_window: QMainWindow) -> None:
        super().__init__()

        self.setSceneRect(0, 0, self.SCENE_WIDTH, self.SCENE_HEIGHT)

        self.parent_window = parent_window
        self.dg = DepGraph()

        # For the selection box
        self.select_rect_item = None
        self.select_start = None
        self.select_end = None

        # Which items we clicked and released on with LMB or RMB
        self.clicked_on_l = None
        self.released_on_1 = None
        self.clicked_on_r = None
        self.released_on_r = None

        # Keeps track of which mouse buttons are currently pressed
        self.mouse_down_l = False
        self.mouse_down_r = False

        # Tracks current placement of dependency arrow
        self.dep_origin = None
        self.dyn_arr = None

        # We can't use QGraphicsItem.data to store these
        # because if you enter an array as a value, it
        # copies it every time you retrieve it, which
        # means you can't modify it
        self.rect_depends_on = {}
        self.rect_influences = {}
        self.rect_arrs_in = {}
        self.rect_arrs_out = {}
        self.rect_risks = {}

    def items_at(self, pos: QPointF) -> list:
        collision_line = self.addLine(QLineF(pos, pos), QPen(Qt.NoPen))
        colliding_items = self.collidingItems(collision_line)
        self.removeItem(collision_line)

        return colliding_items

    def top_rect_at(self, pos: QPointF) -> QGraphicsRectItem:
        top_rect = None
        maxz = float("-inf")
        for item in self.items_at(pos):
            if isinstance(item, QGraphicsRectItem) and item.zValue() > maxz:
                top_rect, maxz = item, item.zValue()

        return top_rect

    def draw_arr(
        self, origin_rect: QGraphicsRectItem, end_pos: QPointF, pen: QPen
    ) -> QGraphicsItemGroup:
        self.del_dyn_arr()

        elbow = None

        arr_tip_pos = end_pos
        point_down = False
        point_up = False

        origin_rect_pos = origin_rect.scenePos()
        start_pos = origin_rect_pos
        arr_start_pos = origin_rect_pos
        left_bound = origin_rect_pos.x()
        right_bound = left_bound + origin_rect.rect().width()
        top_bound = origin_rect_pos.y()
        bot_bound = top_bound + origin_rect.rect().height()

        # Coming out of the sides of the rectangle
        if top_bound < end_pos.y() < bot_bound:
            # Coming out of the left
            if end_pos.x() < left_bound:
                arr_start_pos.setX(left_bound)
            # Coming out of the right
            else:
                arr_start_pos.setX(right_bound)
            arr_start_pos.setY(end_pos.y())
        # Coming out of the top
        elif end_pos.y() <= top_bound:
            arr_start_pos.setY(top_bound)
            arr_start_pos.setX((left_bound + right_bound) / 2)
            elbow = QPointF(start_pos.x(), end_pos.y())
        # Coming out of the bottom
        elif end_pos.y() >= bot_bound:
            arr_start_pos.setY(bot_bound)
            arr_start_pos.setX((left_bound + right_bound) / 2)
            elbow = QPointF(start_pos.x(), end_pos.y())

        moused_over = self.top_rect_at(end_pos)
        if moused_over:
            left_bound = moused_over.scenePos().x()
            right_bound = left_bound + moused_over.rect().width()
            top_bound = moused_over.scenePos().y()
            bot_bound = top_bound + moused_over.rect().height()

            # If we're going straight up or down into the rectangle
            if left_bound < start_pos.x() < right_bound:
                # Coming from above
                if start_pos.y() < end_pos.y():
                    arr_tip_pos.setY(top_bound)
                    point_down = True
                # Coming from below
                else:
                    arr_tip_pos.setY(bot_bound)
                    point_up = True
                arr_tip_pos.setX(start_pos.x())
                elbow = None
            # If we're coming in from the left
            elif start_pos.x() < end_pos.x():
                arr_tip_pos.setX(left_bound)
            # Coming in from the right
            else:
                arr_tip_pos.setX(right_bound)

        if elbow:
            arr_v = self.addLine(QLineF(arr_start_pos, elbow), pen)
            arr_h = self.addLine(QLineF(elbow, arr_tip_pos), pen)
        else:
            # Dummy arrow
            arr_v = self.addLine(QLineF(arr_start_pos, arr_tip_pos), QPen())
            arr_h = self.addLine(QLineF(arr_start_pos, arr_tip_pos), pen)

        # Which way should the arrow point ?
        if point_down:
            arr_bot_l = arr_tip_pos + QPointF(self.ARR_SHORT, -self.ARR_LONG)
            arr_bot_r = arr_tip_pos - QPointF(self.ARR_SHORT, self.ARR_LONG)
        elif point_up:
            arr_bot_l = arr_tip_pos + QPointF(self.ARR_SHORT, self.ARR_LONG)
            arr_bot_r = arr_tip_pos - QPointF(self.ARR_SHORT, -self.ARR_LONG)
        elif end_pos.x() <= start_pos.x():
            arr_bot_l = arr_tip_pos + QPointF(self.ARR_LONG, self.ARR_SHORT)
            arr_bot_r = arr_tip_pos - QPointF(-self.ARR_LONG, self.ARR_SHORT)
        else:
            arr_bot_l = arr_tip_pos - QPointF(self.ARR_LONG, self.ARR_SHORT)
            arr_bot_r = arr_tip_pos + QPointF(-self.ARR_LONG, self.ARR_SHORT)

        arr_tip = self.addPolygon(
            QPolygonF([arr_tip_pos, arr_bot_l, arr_bot_r]), QPen(), QBrush(Qt.black)
        )

        arr = self.createItemGroup([arr_v, arr_h, arr_tip])
        arr.setZValue(-1)

        return arr

    def add_component(self, event: QGraphicsSceneMouseEvent) -> None:
        # Create and add rectangle
        rect_w, rect_h = self.RECT_DIMS
        rect_x = event.scenePos().x() - rect_w // 2
        rect_y = event.scenePos().y() - rect_h // 2
        brush = QBrush(self.parent_window.WPI_RED)

        rect_item = self.addRect(0, 0, rect_w, rect_h, QPen(), brush)
        rect_item.setPos(rect_x, rect_y)
        rect_item.setFlags(QGraphicsItem.ItemIsSelectable)
        rect_item.setData(self.IS_COMPONENT, True)
        self.rect_depends_on[rect_item] = []
        self.rect_influences[rect_item] = []
        self.rect_arrs_in[rect_item] = []
        self.rect_arrs_out[rect_item] = []

        # Create text input box
        comp_name_input = DepQComboBox(rect_item, self, self.parent_window)
        comp_name_input.setCurrentText('')

        # This gives the input box keyboard focus
        QTimer.singleShot(
            0, lambda: comp_name_input.setFocus(Qt.OtherFocusReason)
        )

        # Set up proxy for binding to scene
        input_proxy = QGraphicsProxyWidget(parent=rect_item)
        input_proxy.setWidget(comp_name_input)
    
        # Center input box within rectangle
        input_pos = rect_item.mapFromScene(rect_item.pos())
        input_w = input_proxy.boundingRect().width()
        input_h = input_proxy.boundingRect().height()
        input_pos += QPointF((rect_w - input_w) / 2, (rect_h - input_h) / 2)
        input_proxy.setPos(input_pos)

        # Create risk info label
        comp_risk_label = QLabel(f"Total Risk: {self.dg.DEFAULT_DR:.3f}")
        comp_risk_label.setAlignment(Qt.AlignHCenter)
        comp_risk_label.setMargin(3)

        # Customize font
        font = comp_risk_label.font()
        font.setBold(True)
        comp_risk_label.setFont(font)

        # Customize color palette
        pal = comp_risk_label.palette()
        pal.setColor(QPalette.Window, QColor(0, 0, 0, alpha=0))
        comp_risk_label.setPalette(pal)

        # Create proxy for risk label
        text_proxy = QGraphicsProxyWidget(parent=rect_item)
        text_proxy.setWidget(comp_risk_label)

        # Center in bottom half of rectangle
        text_pos = rect_item.mapFromScene(rect_item.pos())
        text_w = text_proxy.boundingRect().width()
        text_h = text_proxy.boundingRect().height()
        text_pos += QPointF((rect_w - text_w) / 2, (rect_h - text_h) * 3 / 4)
        text_proxy.setPos(text_pos)

        rect_item.setData(self.RISK_LABEL, comp_risk_label)

        self.dg.add_vertex(rect_item)
        self.update_rect_colors()

    def add_AND_gate(self, event: QGraphicsSceneMouseEvent) -> None:
        # Create and add rectangle
        rect_w, rect_h = self.RECT_DIMS
        rect_x = event.scenePos().x() - rect_w // 2
        rect_y = event.scenePos().y() - rect_h // 2
        brush = QBrush(Qt.white)

        rect_item = self.addRect(0, 0, rect_w, rect_h, QPen(), brush)
        rect_item.setPos(rect_x, rect_y)
        rect_item.setFlags(QGraphicsItem.ItemIsSelectable)
        rect_item.setData(self.IS_AND_GATE, True)

        self.rect_depends_on[rect_item] = []
        self.rect_influences[rect_item] = []
        self.rect_arrs_in[rect_item] = []
        self.rect_arrs_out[rect_item] = []
        self.dg.add_AND_gate(rect_item)

        # Create text
        text_widg = QLabel("AND")
        text_widg.setWordWrap(True)
        text_widg.setAlignment(Qt.AlignHCenter)

        # Edit font to be bold and brash
        font = text_widg.font()
        font.setBold(True)
        font.setPointSize(16)
        text_widg.setFont(font)

        # Match background color to rectangle
        pal = text_widg.palette()
        pal.setBrush(QPalette.Window, QBrush(Qt.NoBrush))
        text_widg.setPalette(pal)

        # Set up proxy for binding to scene
        proxy = QGraphicsProxyWidget(parent=rect_item)
        proxy.setWidget(text_widg)

        # Center within rectangle
        text_pos = rect_item.mapFromScene(rect_item.pos())
        text_w = proxy.boundingRect().width()
        text_h = proxy.boundingRect().height()
        text_pos += QPointF((rect_w - text_w) / 2, (rect_h - text_h) / 2)
        proxy.setPos(text_pos)

    def del_select_rect_item(self) -> None:
        # Remove selection box
        if self.select_rect_item:
            self.removeItem(self.select_rect_item)
            self.select_rect_item = None

    def del_dyn_arr(self) -> None:
        if self.dyn_arr:
            self.removeItem(self.dyn_arr)
            self.dyn_arr = None

    def update_rect_colors(self) -> None:
        self.rect_risks = self.dg.get_r_dict()
        for rect in filter(lambda x: x.data(self.IS_COMPONENT), self.items()):
            risk = self.rect_risks[rect]
            brush = rect.brush()
            bcolor = brush.color()
            bcolor.setAlphaF(risk)
            brush.setColor(bcolor)
            rect.setBrush(brush)

            risk_label = rect.data(self.RISK_LABEL)
            risk_label.setText(f"Total Risk: {risk:.3f}")

    # Properly deletes components and AND gates
    def delete_rect(self, rect_item: QGraphicsRectItem) -> None:
        for arr in self.rect_arrs_out[rect_item] + self.rect_arrs_in[rect_item]:
            if arr.scene():
                self.removeItem(arr)
        self.rect_arrs_out[rect_item].clear()
        self.rect_arrs_in[rect_item].clear()

        self.rect_depends_on[rect_item].clear()
        self.rect_influences[rect_item].clear()

        self.dg.delete_vertex(rect_item)
        self.removeItem(rect_item)

    def erase_in_circle(self, pos: QPointF) -> None:
        eraser = self.addEllipse(
            pos.x(), pos.y(),
            self.ERASER_RADIUS,
            self.ERASER_RADIUS,
            QPen(Qt.NoPen),
            QBrush(Qt.NoBrush)
        )

        something_erased = False

        # Clear out edges first to make sure
        # we don't trigger an error trying to
        # delete an edge that's already been deleted
        to_erase = self.collidingItems(eraser)
        for item in to_erase:
            # This will always be true for edges
            if item.data(self.EDGES_VERTICES):
                start, end = item.data(self.EDGES_VERTICES)

                self.rect_depends_on[start].remove(end)
                self.rect_influences[end].remove(start)
                self.dg.delete_edge((start, end))

                self.removeItem(item)
                something_erased = True
        
        # Now deal with components and AND gates
        for item in to_erase:
            if item.data(self.IS_COMPONENT) or item.data(self.IS_AND_GATE):
                self.delete_rect(item)
                something_erased = True

        if something_erased:
            self.update_rect_colors()

        self.removeItem(eraser)
                
    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        match event.button():
            case Qt.LeftButton:
                self.mousePressEventL(event)
            case Qt.RightButton:
                self.mousePressEventR(event)

        super().mousePressEvent(event)

    def mousePressEventL(self, event: QGraphicsSceneMouseEvent) -> None:
        self.mouse_down_l = True
        pos = event.scenePos()
        self.select_start = pos

        # Select whatever we've clicked on
        self.clicked_on_l = self.top_rect_at(pos)
        self.del_select_rect_item()
        if self.clicked_on_l is None:
            # Begin visual selection box
            self.select_rect_item = self.addRect(
                self.select_start.x(),
                self.select_start.y(),
                0,
                0,
                QPen(Qt.black),
                QBrush(Qt.NoBrush),
            )
        else:
            if 1 == len(self.selectedItems()):
                self.clearSelection()

            self.clicked_on_l.setSelected(True)

            # Establish vectors from mouse to items
            for item in self.selectedItems():
                item.setData(self.MOUSE_DELTA, item.scenePos() - pos)

    def mousePressEventR(self, event: QGraphicsSceneMouseEvent) -> None:
        self.mouse_down_r = True

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self.mouse_down_l:
            self.mouseMoveEventL(event)
        if self.dep_origin:
            self.mouseMoveEventWithArrow(event)

        super().mouseMoveEvent(event)

    def mouseMoveEventL(self, event: QGraphicsSceneMouseEvent) -> None:
        if self.dep_origin:
            return

        pos = event.scenePos()

        toolbar = self.parent_window.dep_toolbar
        if self.parent_window.eraser_button == toolbar.selected_tool:
            self.erase_in_circle(pos)
            return

        # Drags objects around, if we should
        if self.clicked_on_l and self.select_start:
            selected = self.selectedItems()
            # Move everything before we redraw arrows
            for item in selected:
                delta = item.data(self.MOUSE_DELTA)
                item.setPos(pos + delta)

            # Redraw arrows
            for item in selected:
                if not isinstance(item, QGraphicsRectItem):
                    continue

                for arr in self.rect_arrs_out[item] + self.rect_arrs_in[item]:
                    if arr.scene():
                        self.removeItem(arr)
                self.rect_arrs_out[item].clear()
                self.rect_arrs_in[item].clear()

                redrawn = set()

                # Redraw the arrows that are going out of the item
                for dependency in self.rect_depends_on[item]:
                    if not dependency.scene():
                        continue

                    dep_center = dependency.scenePos()
                    dep_center += QPointF(
                        dependency.rect().width() / 2, dependency.rect().height() / 2
                    )

                    new_arr = self.draw_arr(item, dep_center, QPen())
                    new_arr.setData(self.EDGES_VERTICES, (item, dependency))

                    self.rect_arrs_out[item].append(new_arr)
                    self.rect_arrs_in[dependency].append(new_arr)

                    redrawn.add((item, dependency))

                # Redraw the arrows that are going into the item
                for influence in self.rect_influences[item]:
                    if not influence.scene() or (influence, item) in redrawn:
                        continue

                    inf_center = influence.scenePos()
                    inf_center += QPointF(
                        influence.rect().width() / 2, influence.rect().height() / 2
                    )

                    item_center = item.scenePos()
                    item_center += QPointF(
                        item.rect().width() / 2, item.rect().height() / 2
                    )

                    new_arr = self.draw_arr(influence, item_center, QPen())
                    new_arr.setData(self.EDGES_VERTICES, (influence, item))

                    self.rect_arrs_out[influence].append(new_arr)
                    self.rect_arrs_in[item].append(new_arr)

            return

        # Adjusts visual selection box
        ax = min(self.select_start.x(), pos.x())
        ay = min(self.select_start.y(), pos.y())

        bx = max(self.select_start.x(), pos.x())
        by = max(self.select_start.y(), pos.y())
        bx -= ax
        by -= ay

        self.del_select_rect_item()
        self.select_rect_item = self.addRect(
            ax, ay, bx, by, QPen(Qt.DashLine), QBrush(Qt.NoBrush)
        )

    def mouseMoveEventWithArrow(self, event: QGraphicsSceneMouseEvent) -> None:
        # Important to note that y-values increase as we go down
        pos = event.scenePos()
        arr_start_pos = self.dep_origin.scenePos()
        arr_start_pos += QPointF(
            self.dep_origin.rect().width() / 2, self.dep_origin.rect().height() / 2
        )

        if self.dep_origin == self.top_rect_at(pos):
            return

        self.dyn_arr = self.draw_arr(self.dep_origin, pos, QPen(Qt.DashLine))

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        match event.button():
            case Qt.LeftButton:
                self.mouseReleaseEventL(event)
            case Qt.RightButton:
                self.mouseReleaseEventR(event)
        
        super().mouseReleaseEvent(event)

    def mouseReleaseEventL(self, event: QGraphicsSceneMouseEvent) -> None:
        self.mouse_down_l = False
        pos = event.scenePos()
        self.clearSelection()

        # Handle single clicks
        self.select_end = pos
        single_click = self.select_start == self.select_end

        # Cancel click if we're just selecting
        # a component from the dropdown
        released_on_l_all = self.items_at(pos)
        for item in released_on_l_all:
            if isinstance(item, QGraphicsProxyWidget) and \
               isinstance(item.widget(), QFrame) and not \
               isinstance(item.widget(), QLabel):
                return

        self.released_on_1 = self.top_rect_at(pos)
        if single_click:
            match self.parent_window.dep_toolbar.selected_tool:
                case self.parent_window.comp_button:
                    if self.released_on_1:
                        self.released_on_1.setSelected(True)
                    else:
                        self.add_component(event)
                case self.parent_window.edge_button:
                    if self.released_on_1:
                        self.mouseReleaseEventEdge(event)
                    else:
                        self.dep_origin = None
                        self.del_dyn_arr()
                case self.parent_window.AND_gate_button:
                    if not self.released_on_1:
                        self.add_AND_gate(event)
                case self.parent_window.eraser_button:
                    self.erase_in_circle(pos)
            return

        if not self.select_rect_item:
            return
        self.del_select_rect_item()

        # Define selection area
        select = QPainterPath()
        rect = QRectF(self.select_start, self.select_end)
        if single_click:
            rect.setBottomRight(self.select_end + QPointF(1, 1))
        select.addRect(QRectF(self.select_start, self.select_end))
        self.setSelectionArea(select)

        # Selections inside one component should be treated as clicks
        if self.clicked_on_l is not None and self.clicked_on_l == self.released_on_1:
            self.clearSelection()
            self.clicked_on_l.setSelected(True)

    def mouseReleaseEventR(self, event: QGraphicsSceneMouseEvent) -> None:
        self.clearSelection()
        self.dep_origin = None
        self.del_dyn_arr()

        pos = event.scenePos()

        self.released_on_r = self.top_rect_at(pos)
        if not self.released_on_r:
            return
        
        global_pos = event.screenPos()
        
        self.context_menu = DepQMenu(self.dg, self.released_on_r, global_pos)

    def mouseReleaseEventEdge(self, event: QGraphicsSceneMouseEvent) -> None:
        self.mouse_down_r = False
        pos = event.scenePos()

        dependent = self.top_rect_at(pos)
        if dependent:
            if not self.dep_origin:
                self.dep_origin = dependent
            else:
                # Don't draw an arrow from a component to itself
                # or redraw an arrow that's already been created
                if (
                    self.dep_origin != dependent
                    and dependent not in self.rect_depends_on[self.dep_origin]
                ):
                    arr_start_pos = self.dep_origin.scenePos()
                    arr_start_pos += QPointF(
                        self.dep_origin.rect().width() / 2,
                        self.dep_origin.rect().height() / 2,
                    )

                    arr_end_pos = dependent.scenePos()
                    arr_end_pos += QPointF(
                        dependent.rect().width() / 2,
                        dependent.rect().height() / 2,
                    )

                    arr = self.draw_arr(self.dep_origin, arr_end_pos, QPen())
                    arr.setData(self.EDGES_VERTICES, (self.dep_origin, dependent))

                    self.rect_arrs_out[self.dep_origin].append(arr)
                    self.rect_arrs_in[dependent].append(arr)

                    self.rect_depends_on[self.dep_origin].append(dependent)
                    self.rect_influences[dependent].append(self.dep_origin)

                    self.dg.add_edge((self.dep_origin, dependent))
                    self.update_rect_colors()

                # Cleanup
                self.dep_origin = None
                self.del_dyn_arr()

    def keyReleaseEvent(self, event) -> None:
        match event.key():
            case Qt.Key_Delete:
                something_deleted = False
                for item in self.selectedItems():
                    self.delete_rect(item)
                    something_deleted = True

                self.dep_origin = None
                self.del_dyn_arr()

                if something_deleted:
                    self.update_rect_colors()

"""

Name: MainWindow
Type: class
Description: MainWindow class that holds all of our functions for the GUI.

"""

class MainWindow(QMainWindow):
    DEFAULT_RISK_THRESHOLD = 1
    CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "data")
    IMAGES_PATH = os.path.join(os.path.dirname(__file__), "images")
    DB_NAME = "part_info.db"
    # These are DataFrame columns
    FAIL_MODE_COLUMNS = (
        "desc",
        "rpn",
        "frequency",
        "severity",
        "detection",
        "lower_bound",
        "best_estimate",
        "upper_bound",
        "mission_time",
    )
    # The types associated with each.
    FAIL_MODE_COLUMN_TYPES = (str, int, int, int, int, float, float, float, float)
    # These are the actual labels to show.
    HORIZONTAL_HEADER_LABELS = [
        "Failure Modes",
        "RPN",
        "Frequency",
        "Severity",
        "Detectability",
        "Lower Bound (LB)",
        "Best Estimate (BE)",
        "Upper Bound (UB)",
        "Mission Time",
    ]
    WPI_RED = QColor(192, 47, 29)

    """

    Name: __init__
    Type: function
    Description: Initializes baseline PyQt5 framework for connecting pieces of the GUI together.

    """

    def __init__(self):
        # These need to match one-to-one
        assert len(self.FAIL_MODE_COLUMNS) == len(self.FAIL_MODE_COLUMN_TYPES)

        super().__init__()

        # Initializes DataFrames.
        self.read_sql()

        self.current_row = 0
        self.current_column = 0
        self.refreshing_table = False
        self.risk_threshold = self.DEFAULT_RISK_THRESHOLD

        self.setWindowTitle("System Reliability Analysis")
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
        # self.main_tool_tab = QWidget()  # Create a new tab
        # self.central_widget.addTab(
        #     self.main_tool_tab, "Main Tool"
        # )  # Add the tab to the QTabWidget
        # self.statistics_tab = QWidget()  # Create a new tab
        # self.central_widget.addTab(
        #     self.statistics_tab, "Statistics"
        # )  # Add the tab to the QTabWidget
        self.dep_tab = QWidget()
        self.central_widget.addTab(self.dep_tab, "Dependencies")
        self.lstm_tab = QWidget()
        self.central_widget.addTab(self.lstm_tab, "lstm")
        self.central_widget.addTab(subtab.NestedTabWidgetS(), "NLP-Unsupervised")
        self.central_widget.addTab(subtab.NestedTabWidgetUnS(), "NLP-Supervised")
        self.central_widget.addTab(similar.SimilarityAnalysisTab(), "NLP-Similarity")

        # self.init_main_tab()
        # self.init_stats_tab()
        self.init_dep_tab()
        self.init_lstm_tab()

        self.qindex = 0
        self.charts = Charts(self)

    def closeEvent(self, event) -> None:
        close_confirm = QMessageBox()
        close_confirm.setWindowTitle("Save and Exit")
        close_confirm.setText("Save before exiting?")
        close_confirm.setStandardButtons(
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
        )
        close_confirm = close_confirm.exec()

        match close_confirm:
            case QMessageBox.Yes:
                self.save_sql()
                event.accept()
            case QMessageBox.No:
                event.accept()
            case QMessageBox.Cancel:
                event.ignore()
            case _:
                event.ignore()

    def init_main_tab(self):
        ### START OF MAIN TAB SETUP ###

        # Create the main layout for the Main Tool tab
        self.main_layout = QHBoxLayout(self.main_tool_tab)

        # Create the left layout for input fields and table
        self.left_layout = QVBoxLayout()

        # Creating label to designate component selection
        self.component_selection = QLabel("Component Selection: ")
        self.left_layout.addWidget(self.component_selection)

        # Create and add the component name dropdown menu
        self.search_and_dropdown_layout_main = QHBoxLayout()

        self.component_name_field = QComboBox(self)
        self.component_name_field.activated.connect(
            lambda: (
                self.component_name_field_stats.setCurrentText(
                    self.component_name_field.currentText()
                ),
                self.update_layout(),
            )
        )
        self.populate_component_dropdown(
            self.component_name_field, self.components["name"]
        )

        self.component_search_field = QLineEdit(self)
        self.component_search_field.setPlaceholderText("Search for a component...")
        self.component_search_field.textChanged.connect(
            self.filter_components(
                self.populate_component_dropdown, self.component_name_field
            )
        )

        self.search_and_dropdown_layout_main.addWidget(self.component_search_field)
        self.search_and_dropdown_layout_main.addWidget(self.component_name_field)

        self.left_layout.addLayout(self.search_and_dropdown_layout_main)

        # Create and add the risk acceptance threshold field
        self.threshold_label = QLabel("Risk Acceptance Threshold:")
        self.threshold_field = QLineEdit()
        self.threshold_field.setText(str(self.DEFAULT_RISK_THRESHOLD))
        self.threshold_field.editingFinished.connect(
            lambda: (self.read_risk_threshold(), self.update_layout())
        )
        self.threshold_field.setToolTip(
            "Enter the maximum acceptable RPN: must be an integer value in [1-1000]."
        )
        self.left_layout.addWidget(self.threshold_label)
        self.left_layout.addWidget(self.threshold_field)

        # Create and add the submit button
        self.submit_button = QPushButton("Reset to Default")
        self.submit_button.clicked.connect(
            lambda: (self.reset_df(), self.update_layout())
        )
        self.left_layout.addWidget(self.submit_button)

        # Create and add the table widget
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(len(self.HORIZONTAL_HEADER_LABELS))
        self.table_widget.setHorizontalHeaderLabels(self.HORIZONTAL_HEADER_LABELS)
        header = self.table_widget.horizontalHeader()
        for i in range(len(self.HORIZONTAL_HEADER_LABELS)):
            item = header.model().headerData(i, Qt.Horizontal)
            headerItem = QTableWidgetItem(str(item))
            headerItem.setTextAlignment(Qt.AlignLeft)
            self.table_widget.setHorizontalHeaderItem(i, headerItem)
        self.table_widget.setColumnWidth(0, 250)  # Failure Mode
        self.table_widget.setColumnWidth(1, 80)  # RPN
        self.table_widget.setColumnWidth(2, 80)  # Frequency
        self.table_widget.setColumnWidth(3, 80)  # Severity
        self.table_widget.setColumnWidth(4, 80)  # Detectability
        self.table_widget.setColumnWidth(5, 110)  # Lower Bound
        self.table_widget.setColumnWidth(6, 110)  # Best Estimate
        self.table_widget.setColumnWidth(7, 110)  # Upper Bound
        self.table_widget.setColumnWidth(8, 110)  # Mission Time
        self.table_widget.verticalHeader().setDefaultSectionSize(32)
        self.table_widget.verticalHeader().setMaximumSectionSize(32)
        self.table_widget.verticalScrollBar().setMaximum(10 * 30)
        self.table_widget.itemChanged.connect(self.table_changed_main)

        self.table_widget.cellClicked.connect(self.cell_clicked)
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
        self.chart_name_field_main_tool.activated.connect(self.generate_main_chart)
        self.right_layout.addWidget(self.chart_name_field_main_tool)

        # Create the matplotlib figure and canvas
        self.main_figure = plt.figure()
        self.canvas = FigureCanvas(self.main_figure)
        self.right_layout.addWidget(self.canvas)

        # Scrolling and zoom in/out functionality
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.right_layout.addWidget(self.toolbar)

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
        # self.prev_button.clicked.connect(self.show_previous)
        self.nav_button_layout.addWidget(self.prev_button)

        # Add spacing between the buttons
        self.nav_button_layout.addSpacing(10)

        self.next_button = QPushButton("Next")
        # self.next_button.clicked.connect(self.show_next)
        self.nav_button_layout.addWidget(self.next_button)

        # Add the nav_button_layout to the left layout
        self.left_layout.addLayout(self.nav_button_layout)

        ### END OF MAIN TAB SETUP ###

    def init_stats_tab(self):
        ### START OF STATISTICS TAB SETUP ###

        # Create main layout
        self.stats_layout = QHBoxLayout(self.statistics_tab)

        # Left layout for component info and table
        self.left_layout_stats = QVBoxLayout()

        # Creating label to designate component selection
        self.component_selection_stats = QLabel("Component Selection: ")
        self.left_layout_stats.addWidget(self.component_selection_stats)

        # Create and add the component name dropdown menu
        self.search_and_dropdown_layout_stats = QHBoxLayout()

        self.component_name_field_stats = QComboBox(self)
        self.component_name_field_stats.addItem("Select a Component")
        self.component_name_field_stats.activated.connect(
            lambda: (
                self.component_name_field.setCurrentText(
                    self.component_name_field_stats.currentText()
                ),
                self.update_layout(),
            )
        )

        self.populate_component_dropdown(
            self.component_name_field_stats, self.components["name"]
        )

        self.component_search_field_stats = QLineEdit(self)
        self.component_search_field_stats.setPlaceholderText(
            "Search for a component..."
        )
        self.component_search_field_stats.textChanged.connect(
            self.filter_components(
                self.populate_component_dropdown, self.component_name_field_stats
            )
        )

        self.search_and_dropdown_layout_stats.addWidget(
            self.component_search_field_stats
        )
        self.search_and_dropdown_layout_stats.addWidget(self.component_name_field_stats)

        self.left_layout_stats.addLayout(self.search_and_dropdown_layout_stats)

        # Create and add the submit button
        self.submit_button_stats = QPushButton("Show Table")
        self.submit_button_stats.clicked.connect(
            lambda: self.populate_table(self.table_widget_stats, self.comp_fails)
        )
        self.left_layout_stats.addWidget(self.submit_button_stats)

        # Create and add the table widget
        self.table_widget_stats = QTableWidget()
        self.table_widget_stats.setColumnCount(len(self.HORIZONTAL_HEADER_LABELS))
        self.table_widget_stats.setHorizontalHeaderLabels(self.HORIZONTAL_HEADER_LABELS)
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
        self.table_widget_stats.cellClicked.connect(self.cell_clicked)
        self.left_layout_stats.addWidget(self.table_widget_stats)

        # Create a QHBoxLayout for the navigation buttons
        self.nav_button_layout_stats = QHBoxLayout()

        # Initialize the selected index and the maximum number of IDs to show
        self.selected_index_stats = 0
        self.max_ids_stats = 10

        # Create and connect the navigation buttons
        self.prev_button_stats = QPushButton("Previous")
        # self.prev_button_stats.clicked.connect(self.show_previous_stats)
        self.nav_button_layout_stats.addWidget(self.prev_button_stats)

        # Add spacing between the buttons
        self.nav_button_layout_stats.addSpacing(10)

        self.next_button_stats = QPushButton("Next")
        # self.next_button_stats.clicked.connect(self.show_next_stats)
        self.nav_button_layout_stats.addWidget(self.next_button_stats)

        # Add the nav_button_layout to the left layout
        self.left_layout_stats.addLayout(self.nav_button_layout_stats)

        # Creating right layout for graphs in stats tab
        self.right_layout_stats = QVBoxLayout()

        # Create label for the graphing
        self.stat_modeling_tag = QLabel("Statistical Modeling: ")
        self.right_layout_stats.addWidget(self.stat_modeling_tag)

        # Create dropdown menu for holding charts we want to give the option of generating
        self.chart_name_field_stats = QComboBox(self)
        self.chart_name_field_stats.addItem("Select a Chart")
        self.chart_name_field_stats.addItem("Weibull Distribution")
        self.chart_name_field_stats.addItem("Rayleigh Distribution")
        self.chart_name_field_stats.addItem("Bathtub Curve")
        self.right_layout_stats.addWidget(self.chart_name_field_stats)

        # Matplotlib canvases with tab widget (hardcoded for one component)
        self.stats_tab = QTabWidget()
        self.stats_tab_canvas1 = FigureCanvas(Figure())
        self.stats_tab.addTab(self.stats_tab_canvas1, "Failure Mode 1")
        self.right_layout_stats.addWidget(self.stats_tab)

        # Create and add the generate chart button
        self.generate_chart_button_stats = QPushButton("Generate Chart")
        self.generate_chart_button_stats.clicked.connect(self.generate_stats_chart)
        self.right_layout_stats.addWidget(self.generate_chart_button_stats)

        # Create and add the download chart button (non-functional)
        self.download_chart_button_stats = QPushButton("Download Chart")
        self.download_chart_button_stats.clicked.connect(
            lambda: self.download_chart(self.stats_tab_canvas1.figure)
        )
        self.right_layout_stats.addWidget(self.download_chart_button_stats)

        # Add left and right layouts to the main layout
        self.stats_layout.addLayout(self.left_layout_stats, 4)
        self.stats_layout.addLayout(self.right_layout_stats, 6)

        ### END OF STATISTICS TAB SETUP ###

    def init_dep_tab(self):
        self.dep_layout = QVBoxLayout(self.dep_tab)

        # Setting up system dependency view
        self.system_vis_scene = DepQGraphicsScene(self)
        self.system_vis_scene.setBackgroundBrush(QBrush(Qt.white, Qt.SolidPattern))

        self.system_vis_view = QGraphicsView(self.system_vis_scene)
        self.system_vis_view.setMouseTracking(True)
        self.system_vis_view.setFrameStyle(QFrame.Panel | QFrame.Plain)
        self.system_vis_view.setLineWidth(2)

        # Right-hand toolbar for selecting mode
        self.dep_toolbar = DepQToolBar()

        # Component button
        self.comp_icon = QIcon(os.path.join(self.IMAGES_PATH, "add_comp_icon.png"))
        self.comp_button = DepQAction(
            self.comp_icon, "Add Component", self.dep_toolbar, self.system_vis_scene, self
        )

        # Edge button
        self.edge_icon = QIcon(os.path.join(self.IMAGES_PATH, "edge_icon.png"))
        self.edge_button = DepQAction(
            self.edge_icon, "Add Edge", self.dep_toolbar, self.system_vis_scene, self
        )

        # AND gate button
        self.AND_gate_icon = QIcon(os.path.join(self.IMAGES_PATH, "and_gate.png"))
        self.AND_gate_button = DepQAction(
            self.AND_gate_icon, "Add AND Gate", self.dep_toolbar, self.system_vis_scene, self
        )

        # Eraser button
        self.eraser_icon = QIcon(os.path.join(self.IMAGES_PATH, "eraser.png"))
        self.eraser_button = DepQAction(
            self.eraser_icon, "Eraser", self.dep_toolbar, self.system_vis_scene, self
        )
        self.eraser_cursor = QCursor(QPixmap(os.path.join(self.IMAGES_PATH, "eraser_cursor.png")))

        # Add widgets separate from setup
        self.dep_layout.addWidget(self.dep_toolbar)
        self.dep_layout.addWidget(self.system_vis_view)

    def init_lstm_tab(self):
        ### START OF lstm TAB SETUP ###

        # Create main layout
        self.lstm_layout = QHBoxLayout(self.lstm_tab)

        # Left layout for component info and table
        self.left_layout_lstm = QVBoxLayout()

        # Creating label to designate component selection
        self.component_selection_lstm = QLabel("Component Selection: ")
        self.left_layout_lstm.addWidget(self.component_selection_lstm)

        # Create and add the component name dropdown menu
        self.search_and_dropdown_layout_lstm = QHBoxLayout()

        self.component_name_field_lstm = QComboBox(self)
        self.component_name_field_lstm.addItem("Select a Component")
        self.component_name_field_lstm.activated.connect(
            lambda: (
                self.component_name_field.setCurrentText(
                    self.component_name_field_lstm.currentText()
                ),
                self.update_layout(),
            )
        )

        self.populate_component_dropdown(
            self.component_name_field_lstm, self.components["name"]
        )

        self.component_search_field_lstm = QLineEdit(self)
        self.component_search_field_lstm.setPlaceholderText("Search for a component...")
        self.component_search_field_lstm.textChanged.connect(
            self.filter_components(
                self.populate_component_dropdown, self.component_name_field_lstm
            )
        )

        self.search_and_dropdown_layout_lstm.addWidget(self.component_search_field_lstm)
        self.search_and_dropdown_layout_lstm.addWidget(self.component_name_field_lstm)

        self.left_layout_lstm.addLayout(self.search_and_dropdown_layout_lstm)

        # Create and add the submit button
        self.submit_button_lstm = QPushButton("Show Table")
        self.submit_button_lstm.clicked.connect(
            lambda: self.populate_table(self.table_widget_lstm, self.comp_fails)
        )
        self.left_layout_lstm.addWidget(self.submit_button_lstm)

        # Create and add the table widget
        self.table_widget_lstm = QTableWidget()
        self.table_widget_lstm.setColumnCount(len(self.HORIZONTAL_HEADER_LABELS))
        self.table_widget_lstm.setHorizontalHeaderLabels(self.HORIZONTAL_HEADER_LABELS)
        self.table_widget_lstm.setColumnWidth(0, 150)  # ID
        self.table_widget_lstm.setColumnWidth(1, 150)  # Failure Mode
        self.table_widget_lstm.setColumnWidth(3, 150)  # RPN
        self.table_widget_lstm.setColumnWidth(4, 150)  # Frequency
        self.table_widget_lstm.setColumnWidth(5, 150)  # Severity
        self.table_widget_lstm.setColumnWidth(6, 150)  # Detectability
        self.table_widget_lstm.setColumnWidth(7, 150)  # Mission Time
        self.table_widget_lstm.setColumnWidth(8, 150)  # Lower Bound
        self.table_widget_lstm.setColumnWidth(9, 150)  # Lower Bound
        self.table_widget_lstm.setColumnWidth(10, 150)  # Best Estimate
        self.table_widget_lstm.verticalHeader().setDefaultSectionSize(32)
        self.table_widget_lstm.verticalHeader().setMaximumSectionSize(32)
        self.table_widget_lstm.verticalScrollBar().setMaximum(10 * 30)
        self.table_widget_lstm.cellClicked.connect(self.cell_clicked)
        self.left_layout_lstm.addWidget(self.table_widget_lstm)

        # Creating right layout for graphs in stats tab
        self.right_layout_lstm = QVBoxLayout()

        # Create label for the graphing
        self.lstm_label_tag = QLabel("LSTM Training: ")
        self.right_layout_lstm.addWidget(self.lstm_label_tag)

        self.start_and_stop_layout = QHBoxLayout()

        # Create and add the submit button
        self.train_button_lstm = QPushButton("Train Model")
        self.start_and_stop_layout.addWidget(self.train_button_lstm)
        self.train_button_lstm.clicked.connect(train_lstm.start_training)

        # Stop training
        self.stop_train_button_lstm = QPushButton("Stop Training")
        self.start_and_stop_layout.addWidget(self.stop_train_button_lstm)
        self.stop_train_button_lstm.clicked.connect(stop_training)

        self.right_layout_lstm.addLayout(self.start_and_stop_layout)

        # Create the matplotlib figure and canvas
        self.loss_fig = plt.figure()
        self.lstm_canvas = FigureCanvas(self.loss_fig)
        self.right_layout_lstm.addWidget(self.lstm_canvas)

        # Scrolling and zoom in/out functionality
        self.lstm_toolbar = NavigationToolbar(self.lstm_canvas, self)
        self.right_layout_lstm.addWidget(self.lstm_toolbar)

        # Add hyperparameter adjusting in gui
        self.hyperparameter_layout_labels = QHBoxLayout()
        self.hyperparameter_layout_boxes = QHBoxLayout()

        self.n_hidden_text = QLabel("N_HIDDEN")
        self.n_hidden_box = QLineEdit()
        self.n_hidden_box.setText(str(train_lstm.N_HIDDEN))
        self.n_hidden_box.editingFinished.connect(self.update_hyperparams)

        self.n_epochs_text = QLabel("N_EPOCHS")
        self.n_epochs_box = QLineEdit()
        self.n_epochs_box.setText(str(train_lstm.N_EPOCHS))
        self.n_epochs_box.editingFinished.connect(self.update_hyperparams)

        self.epoch_size_text = QLabel("EPOCH_SIZE")
        self.epoch_size_box = QLineEdit()
        self.epoch_size_box.setText(str(train_lstm.EPOCH_SIZE))
        self.epoch_size_box.editingFinished.connect(self.update_hyperparams)

        self.learning_rate_text = QLabel("LEARNING_RATE")
        self.learning_rate_box = QLineEdit()
        self.learning_rate_box.setText(str(train_lstm.LEARNING_RATE))
        self.learning_rate_box.editingFinished.connect(self.update_hyperparams)

        self.hyperparameter_layout_labels.addWidget(self.n_hidden_text)
        self.hyperparameter_layout_labels.addWidget(self.n_epochs_text)
        self.hyperparameter_layout_labels.addWidget(self.epoch_size_text)
        self.hyperparameter_layout_labels.addWidget(self.learning_rate_text)

        self.hyperparameter_layout_boxes.addWidget(self.n_hidden_box)
        self.hyperparameter_layout_boxes.addWidget(self.n_epochs_box)
        self.hyperparameter_layout_boxes.addWidget(self.epoch_size_box)
        self.hyperparameter_layout_boxes.addWidget(self.learning_rate_box)

        self.right_layout_lstm.addLayout(self.hyperparameter_layout_labels)
        self.right_layout_lstm.addLayout(self.hyperparameter_layout_boxes)

        self.predict_input_label = QLabel("Predictor")
        self.predict_input_field = QLineEdit()
        self.predict_input_field.setPlaceholderText("Input String...")
        self.predict_input_field.textEdited.connect(self.update_prediction)
        self.predict_output_field = QLineEdit()
        self.predict_output_field.setPlaceholderText("Output Value...")
        
        self.predict_field_layout = QHBoxLayout()
        self.predict_field_layout.addWidget(self.predict_input_field)
        self.predict_field_layout.addWidget(self.predict_output_field)
        
        self.right_layout_lstm.addWidget(self.predict_input_label)
        self.right_layout_lstm.addLayout(self.predict_field_layout)
        
        self.right_layout_lstm.addStretch()

        # Add left and right layouts to the main layout
        self.lstm_layout.addLayout(self.left_layout_lstm, 4)
        self.lstm_layout.addLayout(self.right_layout_lstm, 6)

        ### END OF LSTM TAB SETUP ###

    def update_layout(self):
        self.refreshing_table = True
        self.populate_table(self.table_widget, self.comp_fails)
        self.populate_table(self.table_widget_stats, self.comp_fails)
        self.populate_table(self.table_widget_lstm, self.comp_fails)
        self.generate_main_chart()

        for row in range(len(self.comp_data.index)):
            rpn_item = self.table_widget.item(row, 1)
            if int(rpn_item.text()) > self.risk_threshold:
                rpn_item.setBackground(QColor(255, 102, 102))  # muted red
            else:
                rpn_item.setBackground(QColor(102, 255, 102))  # muted green

        self.refreshing_table = False

    def table_changed_main(self, item):
        if self.refreshing_table:
            return
        self.save_to_df(item)
        self.populate_table(self.table_widget_stats, self.comp_fails)
        self.generate_main_chart()

    """

    Name: generate_chart
    Type: function
    Description: Uses dropdown menu to generate the desired chart in the main tool tab.

    """

    def generate_main_chart(self):
        if "Select a Component" == self.component_name_field.currentText():
            self.chart_name_field_main_tool.setCurrentText("Select a Chart")
            QMessageBox.warning(self, "Error", "Please select a component first.")
            return

        match (self.chart_name_field_main_tool.currentText()):
            case "Bar Chart":
                self.charts.bar_chart()
            case "Pie Chart":
                self.charts.pie_chart()
            case "3D Risk Plot":
                self.charts.plot_3D([self.current_row, self.current_column])
            case "Scatterplot":
                self.charts.scatterplot()
            case "Bubbleplot":
                self.charts.bubble_plot()

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
        self.stats_tab.clear()
        fig1 = stats._bathtub(N, T, t1, t2)

        self.stats_tab_canvas1.figure = fig1
        self.stats_tab_canvas1.figure.tight_layout()
        self.stats_tab_canvas1.draw()

        self.stats_tab.addTab(self.stats_tab_canvas1, "Plot 1")

    """
    
    Name: update_rayleigh_canvas
    Type: function
    Description: Invokes the rayleigh function in stats.py to populate the canvas with a histogram PDF plot.
    
    """

    def update_rayleigh_canvas(self):
        # Clear the existing figures before displaying new ones
        self.stats_tab_canvas1.figure.clear()

        # Clear the existing tabs
        self.stats_tab.clear()

        fig1 = stats._rayleigh(self.values())

        # Update the canvas with the new figures
        self.stats_tab_canvas1.figure = fig1
        self.stats_tab_canvas1.figure.tight_layout()
        self.stats_tab_canvas1.draw()

        # Add tabs after generating the graphs
        self.stats_tab.addTab(self.stats_tab_canvas1, "Plot 1")

    """
    
    Name: updateWeibullCavas
    Type: function
    Description: Invokes the weibull function in stats.py to populate the cavas with a histogram PDF plot.
    
    """

    def update_weibull_canvas(self):
        # Clear the existing figures before displaying new ones
        self.stats_tab_canvas1.figure.clear()

        # Clear the existing tabs
        self.stats_tab.clear()

        fig1 = stats._weibull(self.values())

        # Update the canvas with the new figures
        self.stats_tab_canvas1.figure = fig1
        self.stats_tab_canvas1.figure.tight_layout()
        self.stats_tab_canvas1.draw()

        self.stats_tab.addTab(self.stats_tab_canvas1, "Plot 1")

    """
    Pulls default data from part_info.db and stores it in a pandas DataFrame.
    """

    def read_sql(self) -> None:
        DB_PATH = os.path.abspath(
            os.path.join(self.CURRENT_DIRECTORY, self.DB_PATH, self.DB_NAME)
        )
        if not os.path.isfile(DB_PATH):
            raise FileNotFoundError("could not find database file.")
        self.conn = sqlite3.connect(DB_PATH)
        self.components = pd.read_sql_query("SELECT * FROM components", self.conn)
        self.fail_modes = pd.read_sql_query("SELECT * FROM fail_modes", self.conn)
        self.default_comp_fails = pd.read_sql_query(
            "SELECT * FROM comp_fails", self.conn
        )
        self.comp_fails = pd.read_sql_query("SELECT * FROM local_comp_fails", self.conn)
        # Calculates RPN = Frequency * Severity * Detection
        self.default_comp_fails.insert(
            3,
            "rpn",
            [
                int(row["frequency"] * row["severity"] * row["detection"])
                for _, row in self.comp_fails.iterrows()
            ],
            True,
        )
        self.comp_fails.insert(
            3,
            "rpn",
            [
                int(row["frequency"] * row["severity"] * row["detection"])
                for _, row in self.comp_fails.iterrows()
            ],
            True,
        )
        self.df = pd.merge(
            self.comp_fails, self.components, left_on="comp_id", right_on="id"
        )
        self.df = pd.merge(self.df, self.fail_modes, left_on="fail_id", right_on="id")
        self.df.drop(
            labels=["cf_id", "comp_id", "fail_id", "id_x", "id_y"], axis=1, inplace=True
        )

    def reset_df(self) -> None:
        if not (hasattr(self, "comp_fails") and hasattr(self, "default_comp_fails")):
            return
        self.comp_fails = self.default_comp_fails.copy()

    def read_risk_threshold(self):
        try:
            risk_threshold = float(self.threshold_field.text())
            # error checking for risk value threshold
            if not (1 <= risk_threshold <= 1000):
                error_message = "Error: Please re-enter a risk threshold value between 1 and 1000, inclusive."
                QMessageBox.warning(self, "Value Error", error_message)
        except:
            risk_threshold = self.DEFAULT_RISK_THRESHOLD
        self.risk_threshold = risk_threshold
        return risk_threshold

    """
    Populates a table with failure modes associated with a specific component.
    """

    def populate_table(self, table_widget, data_source) -> None:
        # clear existing table data (does not affect underlying database)
        table_widget.clearContents()

        # retrieve component name from text box
        component_name = self.component_name_field.currentText()

        # Update the column header for "Failure Mode"
        header_labels_static = self.HORIZONTAL_HEADER_LABELS[1:]
        table_widget.setHorizontalHeaderLabels(
            [f"{component_name} Failure Modes"] + header_labels_static
        )

        # Update the maximum number of IDs to show
        self.max_ids = 10

        # drop_duplicates shouldn't be necessary here, 
        # since components are unique. Just in case, though.
        # np.sum is a duct-tapey way to convert to int, 
        # since you can't do that directly
        self.comp_id = np.sum(
            self.components[
                self.components["name"] == component_name
            ].drop_duplicates()["id"]
        )

        self.comp_data = (
            data_source[data_source["comp_id"] == self.comp_id]
            .head(self.max_ids)
            .reset_index(drop=True)
        )

        self.comp_data = pd.merge(
            self.fail_modes, self.comp_data, left_on="id", right_on="fail_id"
        )

        # id column is identical to fail_id column, so we can drop it.
        self.comp_data = self.comp_data.drop(columns="id")

        # Set the row count of the table widget
        table_widget.setRowCount(self.max_ids)

        for row, data in self.comp_data.iterrows():
            for i, key in enumerate(self.FAIL_MODE_COLUMNS):
                table_widget.setItem(row, i, QTableWidgetItem(str(data[key])))

    """
    Records the location of a cell when it's clicked.
    """

    def cell_clicked(self, row, column):
        self.current_row = row
        self.current_column = column

    """
    DONE: get lower bound, geometric mean, and upper bound from dataset, for the component passed in
    """

    def values(self):
        return np.array(
            [
                self.comp_data.at[self.current_row, "lower_bound"],
                self.comp_data.at[self.current_row, "best_estimate"],
                self.comp_data.at[self.current_row, "upper_bound"],
            ]
        )

    # Executes and commits an SQL query on this window's database connection
    def exec_SQL(self, query) -> None:
        self.conn.execute(query)
        self.conn.commit()

    # Saves individual values in the UI to self.comp_fails
    def save_to_df(self, item: QTableWidgetItem) -> None:
        if self.refreshing_table or not hasattr(self, "comp_data"):
            return
        i, j = item.row(), item.column()

        # In case the user is editing a cell below the displayed information.
        if i >= len(self.comp_data.index):
            self.refreshing_table = True
            item.setText("")
            self.refreshing_table = False
            return

        row = self.comp_fails["cf_id"] == self.comp_data.iloc[i]["cf_id"]
        column = self.FAIL_MODE_COLUMNS[j]
        new_val = item.text()

        self.refreshing_table = True
        # Catch invalid entry fields
        if j < 2:
            item.setText(str(self.comp_data.iloc[i][column]))
            self.refreshing_table = False

            QMessageBox.warning(self, "Error", "Cannot edit these fields.")
            return

        try:
            new_val = self.FAIL_MODE_COLUMN_TYPES[j](new_val)
            if not (1 <= new_val <= 10) and (2 <= j <= 4):
                item.setText(str(self.comp_data.iloc[i, j + 3]))
                self.refreshing_table = False

                QMessageBox.warning(
                    self, "Error", "Input must be an integer from 1 to 10, inclusive."
                )
                return
            self.comp_fails.loc[row, column] = new_val
        except ValueError:
            item.setText(str(self.comp_data.iloc[i, j + 3]))
            self.refreshing_table = False

            QMessageBox.warning(self, "Error", "Invalid input for cell type.")
            return

        # If the user is updating FSD, update RPN
        if 2 <= j <= 4:
            new_rpn = int(
                (
                    self.comp_fails.loc[row, "frequency"]
                    * self.comp_fails.loc[row, "severity"]
                    * self.comp_fails.loc[row, "detection"]
                ).iloc[0]
            )
            self.comp_fails.loc[row, "rpn"] = new_rpn
            self.table_widget.setItem(i, 1, QTableWidgetItem(str(new_rpn)))

            rpn_item = self.table_widget.item(i, 1)
            if int(rpn_item.text()) > self.risk_threshold:
                rpn_item.setBackground(QColor(255, 102, 102))  # muted red
            else:
                rpn_item.setBackground(QColor(102, 255, 102))  # muted green
        self.refreshing_table = False

    # Saves local values to the database
    def save_sql(self) -> None:
        for _, row in self.comp_fails.iterrows():
            self.exec_SQL(
                f"""
                UPDATE local_comp_fails
                SET frequency={row["frequency"]},
                    severity={row["severity"]},
                    detection={row["detection"]},
                    lower_bound={row["lower_bound"]},
                    best_estimate={row["best_estimate"]},
                    upper_bound={row["upper_bound"]},
                    mission_time={row["mission_time"]}
                WHERE
                    cf_id={row["cf_id"]}
                """
            )

    """
    Gives user the option to download displayed figure.
    """

    def download_chart(self, figure):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "JPEG (*.jpg);;All Files (*)"
        )
        self.main_figure
        figure.savefig(file_path, format="jpg", dpi=300)

    def filter_components(self, pop_comp_func, field):
        def f(search_query):
            filtered_components = [
                component
                for component in self.components["name"]
                if search_query.lower() in component.lower()
            ]
            pop_comp_func(field, filtered_components)

        return f

    def populate_component_dropdown(self, field, components):
        field.clear()
        field.addItem("Select a Component")
        for name in components:
            field.addItem(name)

    def update_hyperparams(self):
        try:
            train_lstm.N_HIDDEN = int(self.n_hidden_box.text())
            train_lstm.N_EPOCHS = int(self.n_epochs_box.text())
            train_lstm.EPOCH_SIZE = int(self.epoch_size_box.text())
            train_lstm.LEARNING_RATE = float(self.learning_rate_box.text())
        except:
            QMessageBox.warning(
                self, "Cast Error", "Input values aren't able to be cast"
            )
            return

    def update_prediction(self):
        txt = self.predict_input_field.text()
        if(len(txt)==0):
            self.predict_output_field.setText("")
            return
        prediction = train_lstm.predict(self.predict_input_field.text())
        self.predict_output_field.setText(str(prediction.tolist()))

def stop_training():
    train_lstm.stop_training()
    if __name__ == "__main__":
        pool = mp.Pool(NUM_PROCESSES)
        train_lstm.pool = pool


if __name__ == "__main__":
    # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    pool = mp.Pool(NUM_PROCESSES)
    app = QApplication(sys.argv)
    window = MainWindow()
    logger = mp.log_to_stderr(logging.INFO)
    plt.ioff()

    train_lstm.window = window
    train_lstm.pool = pool

    window.show()
    sys.exit(app.exec_())
