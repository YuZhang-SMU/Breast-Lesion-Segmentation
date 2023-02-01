import sys,os
import cv2
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import xlwt, xlrd
import xlutils.copy

class picture(QWidget, QtWidgets.QGraphicsScene):
    def __init__(self, img_path):
        super(picture, self).__init__()
        self.setWindowTitle("point annotations")
        self.path_file = ''
        self.current_1 = [0,0]
        self.current_2 = [0,0]
        self.current_3 = [0,0]
        self.file_name = []
        self.listfile = []
        self.i = 0
        self.j = 0
        self.img_path = img_path

        self.label = QLabel(self)
        self.label.setText("Display image")
        self.label.setFixedSize(500,400)
        self.label.move(150, 150)
        self.label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.label.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(0,0,0,256);font-size:15px;font-weight:bold;font-family:Times New Roman;}"
                                 )

        self.label2 = QLabel(self)
        self.label2.setText("Filename")
        # self.label2.adjustSize()
        self.label2.setFixedSize(700, 30)
        self.label2.move(100, 100)

        self.label2.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(0,0,0,256);font-size:15px;font-weight:bold;font-family:Times New Roman;}"
                                 )

        self.label3 = QLabel(self)
        self.label3.setText("Coordinate1")
        # self.label3.adjustSize()
        self.label3.setFixedSize(100, 30)
        self.label3.move(150, 10)
        self.label3.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(0,0,0,256);font-size:15px;font-weight:bold;font-family:Times New Roman;}"
                                 )

        self.label4 = QLabel(self)
        self.label4.setText("Coordinate2")
        # self.label4.adjustSize()
        self.label4.setFixedSize(100, 30)
        self.label4.move(150, 40)
        self.label4.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(0,0,0,256);font-size:15px;font-weight:bold;font-family:Times New Roman;}"
                                 )

        self.label5 = QLabel(self)
        self.label5.setText("Coordinate3")
        # self.label4.adjustSize()
        self.label5.setFixedSize(100, 30)
        self.label5.move(150, 70)
        self.label5.setStyleSheet("QLabel{background:white;}"
                                  "QLabel{color:rgb(0,0,0,256);font-size:15px;font-weight:bold;font-family:Times New Roman;}"
                                  )

        self.label6 = QLabel(self)
        self.label6.setText("Save")
        self.label6.adjustSize()
        self.label6.setFixedSize(120, 30)
        self.label6.move(10, 200)
        self.label6.setStyleSheet("QLabel{background:white;}"
                                  "QLabel{color:rgb(0,0,0,256);font-size:15px;font-weight:bold;font-family:Times New Roman;}"
                                  )

        btn = QPushButton(self)
        btn.setText("Open dir")
        btn.move(10, 20)
        btn.resize(100, 50)
        btn.setFont(QFont('Times New Roman',14))
        btn.clicked.connect(self.openimage)

        btn = QPushButton(self)
        btn.setText("Save coordinates")
        btn.move(400, 10)
        btn.resize(200, 90)
        btn.setFont(QFont('Times New Roman',15))
        btn.clicked.connect(self.saveimage)

        btn = QPushButton(self)
        btn.setText("Next image")
        btn.move(650, 10)
        btn.resize(200, 90)
        btn.setFont(QFont('Times New Roman',15))
        btn.clicked.connect(self.nextimage)


    def openimage(self):
        self.i = 0
        self.j = 0
        self.count = 0
        self.path_file = QFileDialog.getExistingDirectory(self, "Choose your dir", self.img_path)
        print(self.path_file)
        self.listfile = os.listdir(self.path_file)
        self.file_name = self.listfile[self.i]
        print(self.listfile)
        jpg = QtGui.QPixmap(self.path_file+'/'+self.listfile[self.i])
        print(jpg)
        print(jpg.size)
        self.label.setPixmap(jpg)
        self.label2.setText(self.path_file+'/'+self.listfile[self.i])
        self.label2.adjustSize()
        self.label6.setText('Not saved!!')
        self.label6.adjustSize()
        self.current_1 = [0, 0]
        self.current_2 = [0, 0]
        self.current_3 = [0, 0]


    def saveimage(self):
        self.dataset = self.path_file.split('/')
        self.save_path = os.path.abspath(os.path.join(self.path_file,".."))
        print(self.save_path)
        self.xls_name = self.save_path + '/point_'+str(self.dataset[-1])+'.xls'
        print(1)
        if os.path.exists(self.xls_name):
            self.points = xlrd.open_workbook(self.xls_name)
            self.data = self.points.sheet_by_name('Sheet1')
            self.nrowss = self.data.nrows
            self.ws1 = xlutils.copy.copy(self.points)
            self.table = self.ws1.get_sheet(0)
            if self.current_1[0] != 0 and self.current_2[0] != 0 and self.current_3[0] != 0:
                self.table.write(self.nrowss, 1, label=self.current_1[1] - 150)
                self.table.write(self.nrowss, 2, label=self.current_1[0] - 150)
                self.table.write(self.nrowss, 3, label=self.current_2[1] - 150)
                self.table.write(self.nrowss, 4, label=self.current_2[0] - 150)
                self.table.write(self.nrowss, 5, label=self.current_3[1] - 150)
                self.table.write(self.nrowss, 6, label=self.current_3[0] - 150)
                self.table.write(self.nrowss, 0, label=self.file_name)
                self.count = self.count + 1
            else:
                self.label2.setText('Wrong coordinate! Please ensure to click within the image')
            self.ws1.save(self.xls_name)
        else:
            print(3)
            self.wb = xlwt.Workbook()
            self.ws = self.wb.add_sheet('Sheet1')
            if self.current_1[0] != 0 and self.current_2[0] != 0 and self.current_3[0] != 0:
                self.ws.write(self.count, 1, self.current_1[1] - 150)
                self.ws.write(self.count, 2, self.current_1[0] - 150)
                self.ws.write(self.count, 3, self.current_2[1] - 150)
                self.ws.write(self.count, 4, self.current_2[0] - 150)
                self.ws.write(self.count, 5, self.current_3[1] - 150)
                self.ws.write(self.count, 6, self.current_3[0] - 150)
                self.ws.write(self.count, 0, self.file_name)
                self.count = self.count + 1
            else:
                self.label2.setText('Wrong coordinate! Please ensure to click within the image')
            self.wb.save(self.xls_name)
        self.label6.setText('Saved successfully!')
        self.label6.adjustSize()
        os.makedirs(self.path_file+'_backup', exist_ok=True)
        img = cv2.imread(self.path_file + '/' + self.listfile[self.i], cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(self.path_file+'_backup'+'/' + self.listfile[self.i], img)
        os.remove(self.path_file + '/' + self.listfile[self.i])


    def nextimage(self):
        self.j = 0
        self.current_1 = [0,0]
        self.current_2 = [0,0]
        self.current_3 = [0,0]
        self.label3.setText(f'{self.current_1[0]}, {self.current_1[1]}')
        self.label4.setText(f'{self.current_2[0]}, {self.current_2[1]}')
        self.label5.setText(f'{self.current_3[0]}, {self.current_3[1]}')

        if self.i>len(self.listfile)-2:
            self.label2.setText('This dir has been finished, please choose next dir or close the window to exit.')

        else:
            self.i += 1
            jpg = QtGui.QPixmap(self.path_file+'/'+self.listfile[self.i])
            self.file_name = self.listfile[self.i]
            self.label.setPixmap(jpg)
            self.label2.setText(self.path_file+'/'+self.listfile[self.i])
            self.label2.adjustSize()
            self.label6.setText('Not saved!')
            self.label6.adjustSize()

    def lastimage(self):
        self.count -= 1
        self.j = 0
        self.current_1 = [0, 0]
        self.current_2 = [0, 0]
        self.current_3 = [0, 0]

        self.label3.setText(f'{self.current_1[0]}, {self.current_1[1]}')
        self.label4.setText(f'{self.current_2[0]}, {self.current_2[1]}')
        self.label5.setText(f'{self.current_3[0]}, {self.current_3[1]}')

        self.i -= 1
        jpg = QtGui.QPixmap(self.path_file+'/'+self.listfile[self.i])
        self.file_name = self.listfile[self.i]
        self.label.setPixmap(jpg)
        self.label2.setText(self.path_file+'/'+self.listfile[self.i])
        self.label2.adjustSize()
        self.label6.setText('Saved successfully!')
        self.label6.adjustSize()

    def mousePressEvent(self,evt):
        #evt=QMouseEvent
        if self.j==0:
            self.current_1[0] = evt.pos().x()
            self.current_1[1] = evt.pos().y()
            self.label3.setText(f'{self.current_1[0]}, {self.current_1[1]}')
            self.label3.adjustSize()
            self.j=1
        elif self.j==1:
            self.current_2[0] = evt.pos().x()
            self.current_2[1] = evt.pos().y()
            self.label4.setText(f'{self.current_2[0]}, {self.current_2[1]}')
            self.label4.adjustSize()
            self.j=2
        else:
            self.current_3[0] = evt.pos().x()
            self.current_3[1] = evt.pos().y()
            self.label5.setText(f'{self.current_3[0]}, {self.current_3[1]}')
            self.label5.adjustSize()
            self.j=0


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # replaced with your own image path
    img_path = 'C:/'
    my = picture(img_path)
    my.show()
    sys.exit(app.exec_())