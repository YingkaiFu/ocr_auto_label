import sys
import os
from pathlib import Path
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter,QIcon
from PyQt5.QtWidgets import QApplication, QGraphicsView, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox, QScrollArea, QFileDialog,QGridLayout, QListWidget,QListWidgetItem, QLineEdit
from utils import save_json_files,load_json_files, checkout_det_rec_label
import numpy  as np
from rotate_scene import RotatedRectItem, RotateRectScene


class UniqueListWidget(QListWidget):
    def addItemWithoutDuplicates(self, item_text):
        # 遍历当前所有的条目，检查是否存在相同的文本
        for index in range(self.count()):
            if self.item(index).text() == item_text:
                return False
        
        # 如果没有找到相同的条目，则添加新条目
        super().addItem(QListWidgetItem(item_text))
        return True
    
class ImageItemWidget(QWidget):
    def __init__(self, text, det_color, rec_color, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        self.det_color_label = QLabel()
        self.det_color_label.setFixedSize(10, 10)  # 设置固定大小的颜色块
        self.det_color_label.setStyleSheet(f"background-color: {det_color.name()};")
        self.rec_color_label = QLabel()
        self.rec_color_label.setFixedSize(10, 10)  # 设置固定大小的颜色块
        self.rec_color_label.setStyleSheet(f"background-color: {rec_color.name()};")

        self.text_label = QLabel(text)
        layout.addWidget(self.det_color_label)
        layout.addWidget(self.rec_color_label)
        layout.addWidget(self.text_label)
        layout.setAlignment(Qt.AlignLeft)
        layout.setContentsMargins(5, 0, 0, 0)
        
    def get_text(self):
        """获取 widget 中的文本"""
        return self.text_label.text()

class ImageLabel(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 1600, 800)


        if sys.platform == 'win32':
            import ctypes
            myappid = 'yingkai.auto_label.label_studio.v1.0'  # 任意字符串
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

        self.setWindowTitle("图像标注工具")
        self.setWindowIcon(QIcon("resources/icon.png"))
        
        self.image_files = []
        self.current_image_index = -1
        self.category_color_map = {} 
        self.default_colors = [
            QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255),
            QColor(255, 255, 0), QColor(255, 0, 255), QColor(0, 255, 255),
            QColor(128, 0, 128), QColor(128, 128, 0), QColor(0, 128, 128),
            QColor(255, 165, 0)  # 橙色
        ]  
        self.initUI()


    def add_category(self, category_name):
        """添加一个新的类别并为其分配颜色"""
        if category_name not in self.category_color_map:
            if self.default_colors:  # 如果还有未使用的默认颜色
                color = self.default_colors.pop(0)
            else:  # 如果默认颜色用完了，则随机生成颜色
                color = QColor(
                    np.random.randint(0, 256),
                    np.random.randint(0, 256),
                    np.random.randint(0, 256)
                )
            self.category_color_map[category_name] = color
            self.create_category_button(category_name, color)

    def initUI(self):
        main_layout = QHBoxLayout()

        self.scene = RotateRectScene()

        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.scene.setSceneRect(0, 0, 1280, 720)  # 固定场景大小
        self.view.setMouseTracking(True)
        self.view.setFixedSize(1280, 720)

        left_layout = QVBoxLayout()
        left_layout.setAlignment(Qt.AlignTop)

        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop)

        self.open_folder_button = QPushButton("打开文件夹", self)
        self.open_folder_button.clicked.connect(self.open_folder)

        self.category_groupbox = QGroupBox("类别", self)
        self.category_layout = QGridLayout()
        self.category_groupbox.setLayout(self.category_layout)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.category_groupbox)
        scroll_area.setMinimumHeight(50)
        scroll_area.setMaximumHeight(100)


        self.mode_groupbox = QGroupBox("编辑模式", self)
        self.mode_layout = QHBoxLayout()
        self.mode_groupbox.setLayout(self.mode_layout)

        # 添加模式切换按钮
        self.annotation_mode_button = QPushButton("标注", self)
        self.annotation_mode_button.setCheckable(True)  # 设置为可选中状态
        self.annotation_mode_button.setChecked(True)  # 默认选中标注模式
        self.annotation_mode_button.clicked.connect(self.set_annotation_mode)

        self.text_annotation_mode_button = QPushButton("编辑", self)
        self.text_annotation_mode_button.setCheckable(True)  # 设置为可选中状态
        self.text_annotation_mode_button.clicked.connect(self.set_rec_label_mode)

        self.auto_label_button = QPushButton("自动标注", self)
        self.auto_label_button.setCheckable(True)  # 设置为可选中状态
        self.auto_label_button.clicked.connect(self.set_auto_label_mode)

        self.mode_status_label = QLabel("当前模式: 标注模式", self)

        pos_info_layerout = QHBoxLayout()

        self.current_pos = QLabel("当前鼠标坐标: ", self)
        self.current_color = QLabel("当前鼠标像素: ", self)

        pos_info_layerout.addWidget(self.current_pos)
        pos_info_layerout.addWidget(self.current_color)


        self.scene.mouse_moved.connect(self.update_mouse_position)

         # 连接信号

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.annotation_mode_button)
        button_layout.addWidget(self.text_annotation_mode_button)
        button_layout.addWidget(self.auto_label_button)

        # 添加模式切换按钮和状态标签到主布局
        self.mode_layout.addLayout(button_layout)  # 将按钮布局添加到主布局中
        self.mode_layout.addWidget(self.mode_status_label)  # 状态标签直接添加到主布局


        self.annotation_mode_button.clicked.connect(lambda: self.scene.set_mode("annotation"))
        self.text_annotation_mode_button.clicked.connect(lambda: self.scene.set_mode("rec_label"))
        self.auto_label_button.clicked.connect(lambda: self.scene.set_mode("auto_label"))

        self.text_list_widget = UniqueListWidget(self)
        self.text_list_widget.itemClicked.connect(self.on_text_item_click)
        self.text_list_widget.setMinimumHeight(100)
        self.text_list_widget.setMaximumHeight(300)
        self.text_list_widget.setFocusPolicy(Qt.NoFocus)
        
        self.create_category_buttons()
        # self.add_text_button = QPushButton("新增文本", self)
        # self.add_text_button.clicked.connect(self.show_add_text_dialog)

        self.image_list_widget = QListWidget(self)
        self.image_list_widget.itemClicked.connect(self.on_image_item_click)


        # 自动标注功能UI
        add_label_layerout = QHBoxLayout()

        self.text_edit = QLineEdit(self)  # 文本编辑框
        self.text_edit.setPlaceholderText("请输入预设识别文本")
        # self.image_label.autoRecResultChanged.connect(self.on_auto_rec_result_changed)
        self.confirm_button = QPushButton("确认", self)  # 确认按钮
        self.confirm_button.clicked.connect(self.add_auto_label_text_to_list)

        add_label_layerout.addWidget(self.text_edit)
        add_label_layerout.addWidget(self.confirm_button)


        left_layout.addWidget(self.view)
        left_layout.addLayout(pos_info_layerout)

        right_layout.addWidget(self.open_folder_button)
        right_layout.addWidget(scroll_area)
        right_layout.addWidget(self.mode_groupbox)
        right_layout.addLayout(add_label_layerout)
        right_layout.addWidget(self.text_list_widget)
        right_layout.addWidget(self.image_list_widget)

        main_layout.addLayout(left_layout, 4)
        main_layout.addLayout(right_layout, 1)

        self.setLayout(main_layout)

    def update_mouse_position(self, pos):
        """更新鼠标坐标标签"""
        self.current_pos.setText(f"当前鼠标坐标: ({pos.x():.2f}, {pos.y():.2f})")
        if self.scene.pixmap:
            color = self.scene.pixmap.toImage().pixelColor(int(pos.x()), int(pos.y()))
            self.current_color.setText(f"当前鼠标像素颜色: ({color.red()}, {color.green()}, {color.blue()})")

    def add_text_from_text_input(self):
        text = self.text_edit.text()
        if text:  # 如果用户输入了非空文本
            self.add_text_to_list(text)

    def add_auto_label_text_to_list(self):
        text = self.text_edit.text()
        if text:
            self.add_text_to_list(text)

    def add_text_to_list(self, text):
        """将文本添加到 text_list_widget 中"""

        for index in range(self.text_list_widget.count()):
            item = self.text_list_widget.item(index)
            widget = self.text_list_widget.itemWidget(item)
            if isinstance(widget, QLabel) and widget.text() == text:
                return
        
        # 如果没有找到相同的条目，则创建新条目
        item = QListWidgetItem(self.text_list_widget)
        widget = QLabel(text)
        item.setSizeHint(widget.sizeHint())
        self.text_list_widget.setItemWidget(item, widget)
    

    def set_auto_label_mode(self):
        """切换到自动标注模式"""
        self.auto_label_button.setChecked(True)
        self.annotation_mode_button.setChecked(False)
        self.text_annotation_mode_button.setChecked(False)
        self.mode_status_label.setText("当前模式: 自动标注模式")
        self.scene.drawing_step = 0

    def set_annotation_mode(self):
        """切换到标注模式"""
        self.annotation_mode_button.setChecked(True)
        self.auto_label_button.setChecked(False)
        self.text_annotation_mode_button.setChecked(False)
        self.mode_status_label.setText("当前模式: 标注模式")
        self.scene.drawing_step = 0  # 重置绘制步骤
        self.scene.clear_temp_items()  # 清除临时绘制内容

    def set_rec_label_mode(self):
        """切换到文本标注模式"""
        self.text_annotation_mode_button.setChecked(True)
        self.annotation_mode_button.setChecked(False)
        self.auto_label_button.setChecked(False)
        self.mode_status_label.setText("当前模式: 文本标注模式")
        self.scene.drawing_step = 0
        self.scene.clear_temp_items()  # 清除临时绘制内容


    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder_path:
            self.load_images_from_folder(folder_path)

    def load_images_from_folder(self, folder_path):
        self.image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files.sort()
        self.image_list_widget.clear()

        for img_file in self.image_files:
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            json_file = os.path.join(folder_path, f"{base_name}.json")
            has_det_label, has_rec_label = checkout_det_rec_label(json_file)
            det_color = QColor('green') if has_det_label else QColor('red')
            rec_color = QColor('green') if has_rec_label else QColor('red')
            item = QListWidgetItem(self.image_list_widget)
            widget = ImageItemWidget(os.path.basename(img_file), det_color, rec_color)
            item.setSizeHint(widget.sizeHint())
            self.image_list_widget.setItemWidget(item, widget)

        if self.image_files:
            self.current_image_index = 0
            self.load_image(self.image_files[self.current_image_index])
            self.highlight_current_image()

    def save_current_annotations(self, image_path):
        """保存当前图像的标注信息到JSON文件"""
        json_path = Path(image_path).with_suffix(".json")
        rotate_rect_items = []
        for item in self.scene.items():
            if isinstance(item, RotatedRectItem):
                points = [(point.x(), point.y()) for point in item.current_points]
                rotate_rect_items.append({
                    "points": points,
                    "color": item.brush().color().name(),
                    "category": "ocr",
                    "rec_label": item.rec_item.toPlainText()
                })
        return save_json_files(json_path, Path(image_path), rotate_rect_items)


    def load_image(self, image_path):
        self.scene.load_image(image_path)

        json_path = Path(image_path).with_suffix(".json")
        if os.path.exists(json_path):
            rect_info = load_json_files(json_path, self.category_color_map)
            if rect_info is not None:
                for rects in rect_info:
                    points = rects['points']
                    item = RotatedRectItem(points, rects['color'], rects['category'],rects['rec_label'])
                    item.update_text_position()
                    self.add_text_to_list(rects['rec_label'])
                    self.scene.addItem(item)

    def highlight_current_image(self):
        """高亮当前显示的图像"""
        for index in range(self.image_list_widget.count()):
            item = self.image_list_widget.item(index)
            if index == self.current_image_index:
                item.setSelected(True)
                # 设置选中项的背景颜色为高亮色
            else:
                item.setSelected(False)


    def on_image_item_click(self, item):
        selected_image_name = self.image_list_widget.itemWidget(item).text_label.text()
        selected_image_path = next((img for img in self.image_files if os.path.basename(img) == selected_image_name), None)
        if selected_image_path:
            self.load_image(selected_image_path)
            self.current_image_index = self.image_files.index(selected_image_path)
            self.highlight_current_image()  # 更新高亮


    def on_text_item_click(self, item):
        # 检查是否是重复点击
        widget = self.text_list_widget.itemWidget(item)
        if widget:
            if self.scene.selected_text == widget.text():
                # 取消选中状态
                item.setSelected(False)
                self.selected_text = None  # 清除 selected_text
            else:
                # 更新选中的文本
                self.selected_text =widget.text()
            
            self.scene.selected_text = self.selected_text

    def set_current_category(self, name):
        """设置当前类别的名称和颜色"""
        color = self.category_color_map.get(name, QColor('red'))  # 使用红色作为默认颜色
        self.scene.set_current_category(name, color)
        print(f"选择了类别: {name}, 颜色: {color.name()}")

    def create_category_buttons(self):
        categories = ["ocr"]
        for category in categories:
            self.add_category(category)

    def create_category_button(self, category, color):
        """为指定的类别创建按钮"""
        button = QPushButton(category)
        button.setStyleSheet(f"background-color: {color.name()}; color: white;")
        button.clicked.connect(lambda checked, cat=category: self.set_current_category(cat))
        button.categroy_name = category  # 将类别名称保存为按钮的一个属性

        row = len(self.category_layout) // 2
        col = len(self.category_layout) % 2
        self.category_layout.addWidget(button, row, col)


    def keyPressEvent(self, event):
        if self.image_files and self.current_image_index >= 0:
            current_image_path = self.image_files[self.current_image_index]
            item = self.image_list_widget.item(self.current_image_index)
            widget_item = self.image_list_widget.itemWidget(item)
            n_det, n_rec = self.save_current_annotations(current_image_path)  # 在切换前保存当前图像的标注信息
            det_color = QColor('green') if n_det else QColor('red')
            rec_color = QColor('green') if n_rec else QColor('red')
            widget_item.det_color_label.setStyleSheet(f"background-color: {det_color.name()};")
            widget_item.rec_color_label.setStyleSheet(f"background-color: {rec_color.name()};")

            if event.key() == Qt.Key_A:  # 上一张
                self.scene.clear()
                self.current_image_index = max(0, self.current_image_index - 1)
                self.load_image(self.image_files[self.current_image_index])
                self.highlight_current_image()
            elif event.key() == Qt.Key_D:  # 下一张
                self.scene.clear()
                self.current_image_index = min(len(self.image_files) - 1, self.current_image_index + 1)
                self.load_image(self.image_files[self.current_image_index])
                self.highlight_current_image()

            elif event.key() >= Qt.Key_1 and event.key() <= Qt.Key_9:  # 数字键1-9
                category_keys = list(self.category_color_map.keys())
                if len(category_keys) > 0:
                    index = event.key() - Qt.Key_1  # 将按键转换为索引
                    if 0 <= index < len(category_keys):  # 检查索引是否在范围内
                        selected_category = category_keys[index]
                        self.set_current_category(selected_category)
            else:
                super().keyPressEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('resources/icon.png'))
    window = ImageLabel()
    window.show()
    sys.exit(app.exec_())