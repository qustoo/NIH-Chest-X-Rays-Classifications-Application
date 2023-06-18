import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, \
    QVBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QFileDialog, QMessageBox
from PyQt5.QtGui import QImage
import tensorflow
import cv2
from keras.utils import load_img, img_to_array
from keras.models import load_model
import sys
import pandas as pd
import os

train_set = pd.read_pickle('final_test_set.pkl')
disease_labels = train_set.columns.to_list()[5:]
map_characters = {}
for i in range(0, len(disease_labels)):
    map_characters[i] = disease_labels[i]


class MyLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selectedItem = ""
        self.setAcceptDrops(True)

        self.setAlignment(Qt.AlignCenter)
        self.setText('Example Here')
        self.setStyleSheet('''
        QLabel {
            border: 4px dashed #aaa
        }
        ''')
        self.resize(350, 350)
        self.move(450, 100)

    def setPixmap(self, image):
        super().setPixmap(image)

    def set_image(self, file_path):
        # img = Image.open(file_path)
        # new_image = img.resize(128,128)
        width = self.width()
        height = self.height()
        pixmap = QPixmap(file_path).scaled(width, height)
        self.setPixmap(pixmap)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.set_image(file_path)
            self.selectedItem = file_path
            event.accept()
        else:
            event.ignore()


class ListboxWidget(QListWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        # self.insertItem(0,'red')
        self.resize(300, 600)
        self.move(0, 25)
        # self.doubleClicked.connect(self.show_info_patient)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
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
        self.resize(1200, 600)
        self.setWindowTitle('Check x-ray')
        self.temp_selected_item = None

        self.logo_predicted = QtWidgets.QLabel(self)
        self.logo_predicted.setText("Predicted values")
        # self.logo_predicted.resize(100,50)
        self.logo_predicted.move(100, 0)
        self.lstView = ListboxWidget(self)

        self.logo_actual = QtWidgets.QLabel(self)
        self.logo_actual.setText("Actual disease")
        self.logo_actual.move(1000, 0)
        self.actual_disease = QListWidget(self)
        self.actual_disease.resize(300, 600)
        self.actual_disease.move(900, 25)

        self.model = None
        self.selected_item = ""

        self.logo = MyLabel(self)
        self.logo.setText("\n Drop photo here \n")

        self.btn_load_model = QtWidgets.QPushButton(self)
        self.btn_load_model.setText('Load model')
        self.btn_load_model.setGeometry(300, 500, 200, 50)
        self.btn_load_model.clicked.connect(self.upload_model)

        self.btn_load_image = QtWidgets.QPushButton(self)
        self.btn_load_image.setText("Load a image")
        self.btn_load_image.setGeometry(500, 500, 200, 50)
        self.btn_load_image.clicked.connect(self.load_image)

        self.btn_info = QtWidgets.QPushButton(self)
        self.btn_info.setText('Get prediction')
        self.btn_info.setGeometry(700, 500, 200, 50)
        self.btn_info.clicked.connect(self.get_predicted_result)

    def get_predicted_result(self):
        self.lstView.clear()
        if len(self.logo.selectedItem) == 0 and len(self.selected_item) == 0 and self.model is None:
            messagebox = QMessageBox()
            messagebox.setText("Photo or model not found!")
            messagebox.exec_()
            return
        current_image = self.logo.selectedItem if len(self.selected_item) == 0 else self.selected_item
        print('cur-image', current_image.split('/'))
        result = predict_by_image(current_image, self.model)
        i = 0
        for k, v in result.items():
            self.lstView.insertItem(i, f'{k} = {round(100 * v, 2)}%')
            i += 1
        actual_disease = information_about_patient(current_image.split('/')[-1])
        print(actual_disease)
        for i in range(len(actual_disease)):
            self.actual_disease.insertItem(i, actual_disease[i])

    def upload_model(self):
        path_to_model = QFileDialog.getOpenFileName()
        path_to_model = path_to_model[0]
        print(path_to_model)
        try:
            self.model = load_model(path_to_model)
            mesagebox_ok = QMessageBox()
            mesagebox_ok.setText("Model is uploaded")
            mesagebox_ok.exec_()
        except OSError:
            messagebox = QMessageBox()
            messagebox.setText("Error type of Model!")
            messagebox.exec_()

    def load_image(self):
        path_to_image = QFileDialog.getOpenFileName()
        if path_to_image[0].split('.')[-1] not in ['jpg', 'png']:
            self.selected_item = path_to_image[0]
            mesagebox_ok = QMessageBox()
            mesagebox_ok.setText("Incorrect file")
            mesagebox_ok.exec_()
            return
        else:
            path_to_image = path_to_image[0]
            self.selected_item = path_to_image
            print('self.selected_item', self.selected_item)
            input_img = cv2.imread(self.selected_item)
            enhanced_img = NormalizeContrastSharpnes(input_img)
            h,w,ch = enhanced_img .shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(enhanced_img .data,w,h,bytes_per_line,QImage.Format_RGB888)
            #print(current_img)
            #final_img = NormalizeContrastSharpnes(current_img)
            p = convert_to_Qt_format.scaled(self.logo.width(),self.logo.height())
            result_img  = QPixmap.fromImage(p)
            self.logo.setPixmap(result_img)
            #self.logo.setPixmap(QPixmap(path_to_image).scaled(self.logo.width(), self.logo.height()))


def NormalizeContrastSharpnes(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image
    #kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    #sharped_img = cv2.filter2D(src=enhanced_img, ddepth=-1, kernel=kernel)
    return enhanced_img

def information_about_patient(filename):
    '''
    Принимает путь к изображение и искомую выборку данных
    Возвращает список из всех инфекций грудной клетки
    '''
    new_df = train_set[train_set['filename'] == filename]
    list_disease = str(new_df['disease'].to_list()[0]).split('|')
    id_by_patient = new_df['patientID'].to_list()[0]
    dataframe_by_id = train_set[train_set['patientID'] == id_by_patient].disease
    for disease in dataframe_by_id:
        for i in disease.split('|'):
            if i not in list_disease:
                list_disease.append(i)
    return list_disease




def again_information_about_patient(dict_for_patient, dataframe=train_set):
    # find another disease
    count_snapshot_by_id = dict_for_patient['patientID']

    # transform diseases to list
    dict_for_patient['disease'] = list(dict_for_patient['disease'][0].split('|'))

    if count_snapshot_by_id > 1:
        for diseases in dataframe[dataframe['patientID'] == 13].disease:
            few_disease = diseases.split('|')
            for item_disease in few_disease:
                if item_disease not in dict_for_patient['disease']:
                    dict_for_patient['disease'].append(item_disease)
    dict_for_patient['disease'] = ' '.join(dict_for_patient['disease'])
    return dict_for_patient


def pandas_show():
    pd.set_option('display.width', 500)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.notebook_repr_html', True)


def predict_by_image(path_to_image, uploaded_model):
    '''
    Принимает абсолютный путь к изображение и загруженную модель
    Возвращает словарь болезней с вероятносями по убыванию
    '''
    # image to array
    image = load_img(path_to_image, target_size=(224, 224))
    img_tensor = img_to_array(image)
    img_tensor /= 255
    img_tensor = np.expand_dims(img_tensor, axis=0)

    # prediction
    y_pred = uploaded_model.predict(img_tensor)
    final_dict = {map_characters[i]: y_pred[0][i] for i in range(len(y_pred[0]))}
    return {k: v for k, v in sorted(final_dict.items(), key=lambda item: item[1], reverse=True)}
    #
    # for i in range(len(y_pred[0])):
    #     final_dict[map_characters[i]] = y_pred[0][i]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
