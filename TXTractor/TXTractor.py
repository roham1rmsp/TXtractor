from PyQt5 import QtCore, QtGui, QtWidgets
import os


class Ui_MainWindow(object):

    Camera = os.path.join(os.getcwd(), "TXTractor\camera.jpg")
    Gallery = os.path.join(os.getcwd(), "TXTractor\gallery.jpg")
    
        
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(660, 520)
        MainWindow.setStyleSheet("background: rgb(8, 3, 36)")
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.path = ""
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(460, 210, 51, 51))
        self.pushButton.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.pushButton.setStyleSheet("background: White;\n"
"border: 2px solid White;\n"
"border-radius: 10px")
        self.pushButton.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(self.Camera), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton.setIcon(icon)
        self.pushButton.setIconSize(QtCore.QSize(30, 30))
        self.pushButton.setObjectName("pushButton")
        self.gallery_button = QtWidgets.QPushButton(self.centralwidget, clicked= lambda: self.image_path())
        self.gallery_button.setGeometry(QtCore.QRect(149, 210, 51, 51))
        self.gallery_button.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.gallery_button.setStyleSheet("background: White;\n"
"border: 2px solid White;\n"
"border-radius: 10px")
        self.gallery_button.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(self.Gallery), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.gallery_button.setIcon(icon1)
        self.gallery_button.setIconSize(QtCore.QSize(30, 30))
        self.gallery_button.setObjectName("gallery_button")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(280, 60, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(36)
        self.label.setFont(font)
        self.label.setStyleSheet("color: White\n"
"")
        self.label.setObjectName("label")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(100, 170, 161, 31))
        font = QtGui.QFont()
        font.setPointSize(22)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("color: White")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(430, 170, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(22)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("color:White")
        self.label_4.setObjectName("label_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 660, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def image_path(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName()
        self.path = file_name[0]
    
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Take"))
        self.label_3.setText(_translate("MainWindow", "From Gallery"))
        self.label_4.setText(_translate("MainWindow", "A Photo"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
