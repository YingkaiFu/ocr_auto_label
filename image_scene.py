from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsTextItem
from PyQt5.QtCore import Qt, QPointF, QRectF, QLineF, pyqtSignal
from PyQt5.QtGui import QColor, QPen, QBrush, QPainter, QTransform, QPixmap, QFont, QPainterPath
import numpy as np
import math

class RotatedRectItem(QGraphicsItem):
    itemChanged = pyqtSignal(object)
    
    def __init__(self, points, color=QColor(255,0,0,50), label=""):
        super().__init__()
        self.points = points  # 四个顶点坐标（顺时针）
        self.normal_color = color
        self.selected_color = QColor(0, 255, 0)
        self.control_size = 8
        self.is_selected = False
        self.label = label
        self.text_item = QGraphicsTextItem(label, self)
        
        self.setFlags(QGraphicsItem.ItemIsMovable | 
                     QGraphicsItem.ItemIsSelectable |
                     QGraphicsItem.ItemSendsGeometryChanges)
        
        self.create_control_points()
        self.update_text_position()

    def boundingRect(self):
        return self.shape().boundingRect()

    def shape(self):
        path = QPainterPath()
        path.addPolygon(self.polygon())
        return path

    def paint(self, painter, option, widget):
        pen = QPen(self.selected_color if self.is_selected else self.normal_color, 2)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(255,0,0,50)))
        painter.drawPolygon(self.polygon())

        # 绘制控制点
        if self.is_selected:
            painter.setBrush(QBrush(Qt.blue))
            for cp in self.control_points:
                painter.drawEllipse(cp, self.control_size/2, self.control_size/2)

    def polygon(self):
        return self.points

    def create_control_points(self):
        self.control_points = [
            self.points[0],  # 左上
            (self.points[0] + self.points[1])/2,  # 上中
            self.points[1],  # 右上
            (self.points[1] + self.points[2])/2,  # 右中
            self.points[2],  # 右下
            (self.points[2] + self.points[3])/2,  # 下中
            self.points[3],  # 左下
            (self.points[3] + self.points[0])/2   # 左中
        ]

    def update_text_position(self):
        center = self.polygon().boundingRect().center()
        self.text_item.setPos(center - self.text_item.boundingRect().center())

    def mousePressEvent(self, event):
        self.start_pos = event.pos()
        self.start_points = self.points.copy()
        self.selected_control = self.find_control_point(event.pos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.selected_control is not None:
            delta = event.pos() - self.start_pos
            self.adjust_shape(delta)
            self.start_pos = event.pos()
            self.start_points = self.points.copy()
        else:
            super().mouseMoveEvent(event)

    def adjust_shape(self, delta):
        # 根据拖拽的控制点调整形状
        idx = self.control_points.index(self.selected_control)
        new_point = self.start_points[idx] + delta
        
        if idx % 2 == 0:  # 角点
            opposite_idx = (idx + 4) % 8
            center = (self.points[idx] + self.points[opposite_idx])/2
            self.points[idx] = new_point
            self.points[(idx+1)%4] = self.rotate_point(self.points[(idx+1)%4], center, delta)
            self.points[(idx-1)%4] = self.rotate_point(self.points[(idx-1)%4], center, delta)
        else:  # 边中点
            # 计算边向量并调整对应边
            pass
        
        self.create_control_points()
        self.update_text_position()
        self.itemChanged.emit(self)
        self.update()

    def rotate_point(self, point, center, delta):
        angle = math.atan2(delta.y(), delta.x())
        dist = math.hypot(delta.x(), delta.y())
        v = point - center
        r = math.hypot(v.x(), v.y())
        theta = math.atan2(v.y(), v.x()) + angle
        return center + QPointF(r*math.cos(theta), r*math.sin(theta))

    def find_control_point(self, pos):
        for cp in self.control_points:
            if QLineF(pos, cp).length() < self.control_size:
                return cp
        return None

    def set_label(self, text):
        self.text_item.setPlainText(text)
        self.update_text_position()

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedChange:
            self.is_selected = bool(value)
            self.update()
        return super().itemChange(change, value)

class AnnotationScene(QGraphicsScene):
    rectCreated = pyqtSignal(RotatedRectItem)
    
    def __init__(self):
        super().__init__()
        self.current_rect = None
        self.mode = "draw"
        self.points_collected = []

    def mousePressEvent(self, event):
        if self.mode == "draw":
            pos = event.scenePos()
            if len(self.points_collected) < 2:
                self.points_collected.append(pos)
                if len(self.points_collected) == 2:
                    # 创建初始矩形
                    self.current_rect = RotatedRectItem([pos, pos, pos, pos])
                    self.addItem(self.current_rect)
            elif len(self.points_collected) == 2:
                # 确定高度
                self.finalize_rectangle(pos)
                self.points_collected = []
        else:
            super().mousePressEvent(event)

    def finalize_rectangle(self, pos):
        p1, p2 = self.points_collected
        vec = p2 - p1
        normal = QPointF(-vec.y(), vec.x()).normalized()
        height = QLineF(pos, p2).length()
        p3 = p2 + normal * height
        p4 = p1 + normal * height
        
        self.current_rect.points = [p1, p2, p3, p4]
        self.current_rect.create_control_points()
        self.rectCreated.emit(self.current_rect)
        self.current_rect = None

class AnnotationView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = AnnotationScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.zoom_factor = 1

        # 连接信号
        self.scene.rectCreated.connect(self.handle_new_rect)

    def wheelEvent(self, event):
        factor = self.zoom_factor ** (event.angleDelta().y() / 240)
        self.scale(factor, factor)

    def handle_new_rect(self, rect_item):
        # 执行OCR识别等后续处理
        pass

    def load_image(self, image_path):
        self.scene.clear()
        pixmap = QPixmap(image_path)
        self.scene.addPixmap(pixmap)
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)