import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGraphicsView, 
                            QVBoxLayout, QWidget, QLabel, QPushButton)
from PyQt5.QtCore import Qt, QPointF, QLineF, pyqtSignal,QLine, QPoint,QRectF
from PyQt5.QtGui import QColor, QPen, QBrush, QPolygonF, QPainter, QTransform, QTextCursor, QPainterPath, QFont
from PyQt5.QtWidgets import QGraphicsPolygonItem, QGraphicsTextItem, QGraphicsItem, QGraphicsScene, QGraphicsLineItem, QGraphicsPathItem, QGraphicsRectItem,QGraphicsItemGroup
from PyQt5.QtGui import QPixmap, QPen,QPainter
from ocr_rec_onnx import rec_infer_roi




def projection_on_c(a, b):
    c = np.array([-a[1], a[0]])  # 垂直于 a 的向量
    
    if np.dot(b, c) < 0:  # 如果 b 和 c 方向相反
        c = -c  

    projection = (np.dot(b, c) / np.dot(c, c)) * c
    
    return projection

class CustomTextItem(QGraphicsTextItem):
    def __init__(self, text):
        super().__init__(text)
        self.font = QFont()
        self.font.setBold(True)
        self.setFont(self.font)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # 当按下回车键时，保存编辑并退出编辑模式
            self.setTextInteractionFlags(Qt.NoTextInteraction)
            self.clearFocus()
        else:
            super().keyPressEvent(event)


class RotatedRectItem(QGraphicsPolygonItem):
    def __init__(self, points, color, category, rec_label=""):
        super().__init__(QPolygonF(points))
        
        # 创建多边形
        self.setPen(QPen(color, 2))

        
        # 创建箭头线段
        self.direction_edge = ArrowLineItem(points[0], points[1])
        self.direction_edge.setParentItem(self)
        self.direction_edge.setPen(QPen(color, 2))
        
        # 创建文本
        self.rec_item = CustomTextItem(rec_label)
        self.rec_item.setDefaultTextColor(color)
        self.rec_item.setScale(2)
        self.rec_item.setParentItem(self)
        
        # 更新文本位置
        self.update_text_position()
        
        # 启用悬停事件
        self.setAcceptHoverEvents(True)
        
        # 存储原始点
        self.original_points = points
        self.last_pos = self.scenePos()
        self.new_pos = self.scenePos()
        self.current_points = self.polygon()

    def update_text_position(self):
        polygon_points = self.polygon()  # 获取多边形点集
        if len(polygon_points) < 2:
            return
        
        p1 = QPointF(polygon_points[0])  # 创建 p1 的副本
        p2 = QPointF(polygon_points[1])  # 创建 p2 的副本
        offset_point = QPointF(-10, -35)
        text_pos = p1 + offset_point
        line = QLineF(p1, p2)
        angle = line.angle()
        self.rec_item.setPos(text_pos)
        self.rec_item.setRotation(-angle)

    def set_focus_on_edit(self):
        self.rec_item.setTextInteractionFlags(Qt.TextEditorInteraction)
        self.rec_item.setFocus()
        # 移动光标到文本末尾
        cursor = self.rec_item.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.rec_item.setTextCursor(cursor)

    def mouseDoubleClickEvent(self, event):
        # 设置文本为可编辑状态
        self.set_focus_on_edit()
        super().mouseDoubleClickEvent(event)

    def save_text_edit_changes(self, event):
        # 当用户完成编辑时（例如，失去焦点），保存更改并恢复文本交互标志
        self.rec_item.setTextInteractionFlags(Qt.NoTextInteraction)
        self.update()  # 更新显示
        # 调用原来的focusOutEvent
        QGraphicsTextItem.focusOutEvent(self.rec_item, event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            for index, point in enumerate(self.original_points):
                # 将每个原始点加上位置变化量，并添加到updated_points
                updated_point = point + value
                self.current_points[index] = updated_point

        return super().itemChange(change, value)
    
    def set_focus_on_edit(self):
        self.rec_item.setTextInteractionFlags(Qt.TextEditorInteraction)
        self.rec_item.setFocus()
        # 移动光标到文本末尾
        cursor = self.rec_item.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.rec_item.setTextCursor(cursor)

    def mouseDoubleClickEvent(self, event):
        # 设置文本为可编辑状态
        self.set_focus_on_edit()
        super().mouseDoubleClickEvent(event)

    def save_text_edit_changes(self, event):
        # 当用户完成编辑时（例如，失去焦点），保存更改并恢复文本交互标志
        self.rec_item.setTextInteractionFlags(Qt.NoTextInteraction)
        self.update()  # 更新显示
        # 调用原来的focusOutEvent
        QGraphicsTextItem.focusOutEvent(self.rec_item, event)
        
class ArrowLineItem(QGraphicsLineItem):
    def __init__(self, x, y, color=Qt.red):
        super().__init__()
        self.setPen(QPen(color, 2))

        self.setLine(QLineF(x,y))
        self.arrow_size = 10  # 箭头大小
        self.arrow_item = None  # 用于存储箭头图形项
        self.update_arrow()  # 初始化时更新箭头

    def update_arrow(self):
        """更新箭头的位置和方向"""
        if self.arrow_item:
            self.scene().removeItem(self.arrow_item)  # 移除旧的箭头

        line = self.line()
        start_point = line.p1()
        end_point = line.p2()

        # 计算箭头的角度
        angle = np.arctan2(end_point.y() - start_point.y(), end_point.x() - start_point.x())
        arrow_len = self.arrow_size
        arrow_angle = np.pi / 6  # 箭头的角度宽度

        # 计算箭头的三个点
        arrow_point1 = QPointF(end_point.x() - arrow_len * np.cos(angle + arrow_angle),
                               end_point.y() - arrow_len * np.sin(angle + arrow_angle))
        arrow_point2 = QPointF(end_point.x() - arrow_len * np.cos(angle - arrow_angle),
                               end_point.y() - arrow_len * np.sin(angle - arrow_angle))

        # 创建箭头的多边形
        arrow_polygon = QPolygonF([end_point, arrow_point1, arrow_point2])
        self.arrow_item = QGraphicsPolygonItem(arrow_polygon, self)
        self.arrow_item.setPen(self.pen())
        self.arrow_item.setBrush(self.pen().color())  # 填充箭头颜色

    def setLine(self, line):
        """重写 setLine 方法，更新箭头"""
        super().setLine(line)
        if self.scene():  # 如果已经添加到场景中
            self.update_arrow()

def item_reset_flags(item: QGraphicsItem):
    item.setFlag(QGraphicsItem.ItemIsMovable, False)
    item.setFlag(QGraphicsItem.ItemIsSelectable, False)
    item.setFlag(QGraphicsItem.ItemSendsGeometryChanges, False)

def item_set_flags(item: QGraphicsItem):
    item.setFlags(QGraphicsPolygonItem.ItemIsMovable | QGraphicsPolygonItem.ItemIsSelectable| QGraphicsPolygonItem.ItemSendsGeometryChanges)


class RotateRectScene(QGraphicsScene):
    drawing_finished = pyqtSignal(list)  # 参数为四个顶点坐标
    mouse_moved = pyqtSignal(QPointF)

    def __init__(self):
        super().__init__()
        self.drawing_step = 0  # 0-未绘制 1-第一个点 2-第二个点 3-第三个点
        self.temp_points = []
        self.current_color = QColor(255, 0, 0)
        self.current_category = "ocr"
        self.rec_label = ""
        self.temp_lines = []
        self.current_image = None
        self.temp_preview = None
        self.current_mode = "annotation"  # 当前模式，默认为标注模式
        self.selected_text = ""
        self.pixmap = None


    def set_current_category(self, category, color):
        self.current_category = category
        self.current_color = color

    def load_image(self, image_path):
        self.current_image = image_path
        self.pixmap =  QPixmap(image_path)
        self.addPixmap(self.pixmap)
    
    def mousePressEvent(self, event):
        if self.current_mode == "annotation":
            if event.button() == Qt.LeftButton:
                item = self.itemAt(event.scenePos(), self.views()[0].transform())
                if isinstance(item, RotatedRectItem):
                    item_reset_flags(item)
                self.handle_left_click(event.scenePos())
            elif event.button() == Qt.RightButton:
                self.handle_right_click()
        elif self.current_mode == "auto_label":
            if event.button() == Qt.LeftButton:
                item = self.itemAt(event.scenePos(), self.views()[0].transform())
                if isinstance(item, RotatedRectItem):
                    item_reset_flags(item)
                    roi = np.array([[point.x(), point.y()] for point in item.current_points])
                    infer_rec_label = rec_infer_roi(self.current_image, roi)
                    item.rec_item.setDefaultTextColor(self.current_color)
                    item.rec_item.setPlainText(infer_rec_label[0])
                    item.set_focus_on_edit()
        else:
            if event.button() == Qt.LeftButton:
                item = self.itemAt(event.scenePos(), self.views()[0].transform())
                if isinstance(item, RotatedRectItem):
                    item_set_flags(item)

                    if self.selected_text:
                        item.rec_item.setPlainText(self.selected_text)
        if event.button() == Qt.LeftButton:
            item = self.itemAt(event.scenePos(), self.views()[0].transform())
            if isinstance(item, RotatedRectItem):
                self.selected_item = item  # 保存当前选中的 item
            else:
                self.selected_item = None
        elif event.button() == Qt.RightButton:
            self.selected_item = None

        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        # 监听 Delete 键的按下事件
        if event.key() == Qt.Key_Delete:
            if hasattr(self, 'selected_item') and self.selected_item is not None:
                self.removeItem(self.selected_item)  # 删除选中的 item
                self.selected_item = None  # 清空选中的 item
        super().keyPressEvent(event)

    def set_mode(self, mode):
        """设置当前模式"""
        self.current_mode = mode

    def handle_left_click(self, pos):
        if self.drawing_step == 0:
            self.start_drawing(pos)
        elif self.drawing_step == 1:
            self.set_base_edge(pos)
        elif self.drawing_step == 2:
            self.complete_rectangle(pos)

    def handle_right_click(self):
        if self.drawing_step > 0:
            self.cancel_drawing()

    def start_drawing(self, pos):
        self.drawing_step = 1
        self.temp_points = [pos]
        arrow_line = ArrowLineItem(pos,pos)
        arrow_line.setPen(QPen(Qt.red, 2))
        self.addItem(arrow_line)
        self.temp_lines.append(arrow_line)

    def calculate_points(self,pos):
        p1 = self.temp_points[0]
        p2 = self.temp_points[1]

        vector_first = np.array(p2) - np.array(p1)
        vector_second = np.array(pos) - np.array(p2)

        vector_first_np = np.array([vector_first.x(), vector_first.y()])
        vector_second_np = np.array([vector_second.x(), vector_second.y()])
        vector_proj = projection_on_c(vector_first_np, vector_second_np)
        vector_proj_qt = QPoint(int(vector_proj[0]), int(vector_proj[1]))
        p3 = p2 + vector_proj_qt
        p4 = p1 + vector_proj_qt
        return p3,p4
    
    def set_base_edge(self, pos):
        self.drawing_step = 2
        self.temp_points.append(pos)
        self.temp_lines[0].setLine(QLineF(self.temp_points[0], pos))
        line = self.addLine(QLineF(pos, pos), QPen(Qt.red, 2))
        self.temp_lines.append(line)
        line = self.addLine(QLineF(pos, pos), QPen(Qt.red, 2))
        self.temp_lines.append(line)
        line = self.addLine(QLineF(pos, pos), QPen(Qt.red, 2))
        self.temp_lines.append(line)

    def create_arrow_path(self, start_point, end_point):
        path = QPainterPath()
        path.moveTo(start_point)
        path.lineTo(end_point)
        
        angle = np.arctan2(end_point.y() - start_point.y(), end_point.x() - start_point.x())
        arrow_len = 10  # 箭头大小
        arrow_angle = np.pi / 6  # 箭头的角度宽度
        
        # 计算箭头两个点的位置
        arrow_point1 = QPointF(end_point.x() - arrow_len * np.cos(angle + arrow_angle),
                            end_point.y() - arrow_len * np.sin(angle + arrow_angle))
        arrow_point2 = QPointF(end_point.x() - arrow_len * np.cos(angle - arrow_angle),
                            end_point.y() - arrow_len * np.sin(angle - arrow_angle))
        
        path.lineTo(arrow_point1)
        path.moveTo(end_point)
        path.lineTo(arrow_point2)
        
        return path


    def complete_rectangle(self, pos):
        self.drawing_step = 0
        p3, p4 = self.calculate_points(pos)
        final_points = [self.temp_points[0], self.temp_points[1], p3, p4]
        
        # 创建矩形项
        rect_item = RotatedRectItem(final_points, self.current_color, self.current_category)
        self.addItem(rect_item)
                
        self.clear_temp_items()
        self.drawing_finished.emit([(p.x(), p.y()) for p in final_points])

    def cancel_drawing(self):
        self.drawing_step = 0
        self.temp_points = []
        self.clear_temp_items()

    def clear_temp_items(self):
        for line in self.temp_lines:
            self.removeItem(line)
        self.temp_lines = []

    def mouseMoveEvent(self, event):
        mouse_pos = event.scenePos()
        self.mouse_moved.emit(mouse_pos)
        if self.current_mode == "annotation":
            if self.drawing_step == 1:
                self.update_base_edge_preview(event.scenePos())
            elif self.drawing_step == 2:
                self.update_height_preview(event.scenePos())
        super().mouseMoveEvent(event)
        
    def update_base_edge_preview(self, pos):
        if self.temp_lines:
            self.temp_lines[0].setLine(QLineF(self.temp_points[0], pos))

    def update_height_preview(self, pos):
        if len(self.temp_lines) > 1:
            self.temp_lines[0].setLine(QLineF(self.temp_points[0], self.temp_points[1]))
            p3,p4 = self.calculate_points(pos)
            self.temp_lines[1].setLine(QLineF(self.temp_points[1], p3))
            self.temp_lines[2].setLine(QLineF(p3, p4))

            self.temp_lines[3].setLine(QLineF(p4, self.temp_points[0]))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("旋转矩形标注示例")
        
        # 创建场景和视图
        self.scene = RotateRectScene()


        self.view = QGraphicsView(self.scene)
        # self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setAlignment(Qt.AlignCenter)
        self.scene.setSceneRect(0, 0, 1280, 720)  # 固定场景大小
        self.view.setMouseTracking(True)
        # self.view.setFixedSize(1280, 720)
        # 创建状态标签
        self.status_label = QLabel("准备就绪")
        self.clear_btn = QPushButton("清空所有标注")
        self.clear_btn.clicked.connect(self.clear_all)
        
        # 设置布局
        layout = QVBoxLayout()
        layout.addWidget(self.view)
        # layout.addWidget(self.status_label)
        # layout.addWidget(self.clear_btn)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        # 连接信号
        self.scene.drawing_finished.connect(self.on_drawing_finished)

    def on_drawing_finished(self, points):
        self.status_label.setText(f"已创建旋转矩形，顶点坐标：{points}")

    def clear_all(self):
        self.scene.clear()
        self.status_label.setText("已清空所有标注")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())