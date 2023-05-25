from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, \
    QVBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QFileDialog
import sys


class MyLabel(QLabel):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self.setAlignment(Qt.AlignCenter)
        self.setText('Example Here')
        self.resize(200,200)
        self.move(800,0)


class ListboxWidget(QListWidget):


    def __init__(self,parent = None):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self.resize(600,600)

        self.doubleClicked.connect(self.show_name)


    def show_name(self):
        selected_item = self.currentItem()
        print(selected_item.text())
    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()
    def dragMoveEvent(self,event):
        if event.mimeData().hasImage:
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self,event):
        if event.mimeData().hasImage:
            event.setDropAction(Qt.CopyAction)

            event.accept()
            print()
            images = []
            for image in event.mimeData().urls():
                if image.isLocalFile():
                    images.append(str(image.toLocalFile()))
                else:
                    images.append(str(image.toString()))
            self.addItems(images)
        else:
            event.ignore()



class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1200,600)
        self.setWindowTitle('Check x-ray')

        self.lstView = ListboxWidget(self)

        self.logo = MyLabel(self)

        self.btn = QtWidgets.QPushButton(self)
        self.btn.setText('Get Info')
        self.btn.setGeometry(800,400,200,50)
        self.btn.clicked.connect(lambda: print(self.getSelectedItem()))

        self.btn_info = QtWidgets.QPushButton(self)
        self.btn_info.setText('Get info about photo')
        self.btn_info.setGeometry(800,300,200,50)

    def getSelectedItem(self):
        item = QListWidgetItem(self.lstView.currentItem())
        return item.text()




        # self.flag_added_text = False

        # self.logo = QtWidgets.QLabel(self)
        # self.logo.resize(500,500)
        # self.logo.setText('')
        # self.logo.setPixmap(QPixmap('test.jpg'))
        # self.logo.setStyleSheet("border :3px solid black;")
        # self.logo.move(100, 100)
        #
        #
        #
        #
        # self.btn = QtWidgets.QPushButton(self)
        # self.btn.move(70, 50)
        # self.btn.setText("Нажми")
        # self.btn.setFixedWidth(200)
        # self.btn.clicked.connect(self.upload_image)

        self.show()

    def upload_image(self):
        filename = QFileDialog.getOpenFileName()
        path = filename[0]
        pixmap = QPixmap(path[0])
        self.logo.setPixmap(pixmap)

        print(path)

    def add_label(self):
        if not self.flag_added_text:
            self.new_text.setText('second')
            self.new_text.move(40, 50)
            self.new_text.adjustSize()
            self.flag_added_text = True

        else:
            self.flag_added_text = False
            self.new_text.setText('')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()

    sys.exit(app.exec_())
