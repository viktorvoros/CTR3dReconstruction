import sys
from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget)
from PyQt5.QtWidgets import QFileDialog
import os

class CTR_GUI(QDialog):
    def __init__(self, parent=None):
        super(CTR_GUI, self).__init__(parent)

        self.originalPalette = QApplication.palette()

        self.createTopLeftGroupBox()
        self.createTopRightGroupBox()
        # self.createBottomLeftTabWidget()
        self.createBottomRightGroupBox()

        mainLayout = QGridLayout()

        mainLayout.addWidget(self.topLeftGroupBox, 1, 0)
        mainLayout.addWidget(self.topRightGroupBox, 1, 1)
        # mainLayout.addWidget(self.bottomLeftTabWidget, 2, 0)
        mainLayout.addWidget(self.bottomRightGroupBox, 2, 1)
        mainLayout.setRowStretch(1, 1)
        mainLayout.setRowStretch(2, 1)
        mainLayout.setColumnStretch(1, 1)
        mainLayout.setColumnStretch(1, 1)
        self.setLayout(mainLayout)

        self.setWindowTitle("CTR tracking GUI")
        self.show()
        QApplication.exec_()

    def createTopLeftGroupBox(self):
        self.topLeftGroupBox = QGroupBox("User input", self)

        self.numTubeSel = QSpinBox()
        self.numTubeSel.setValue(3)
        self.numTubeSel.setMinimum(2)
        self.numTubeSel.setMaximum(5)
        self.numTubeSel.valueChanged.connect(self.numTube)
        self.nTube = self.numTubeSel.value()

        self.numLabel = QLabel("Number of tubes:", self)
        self.numLabel.setBuddy(self.numTubeSel)
        c = self.numTubeSel.value()
        # print('Number of tubes: ', c)

        self.dataLabel = QLabel("Calibration data:", self)
        self.browse = QPushButton("Browse..", self)
        self.browse.setDefault(True)
        self.browse.clicked.connect(self.browseClick)

        self.source = QLabel("", self)
        self.src = str(self.source.text())


        layout = QVBoxLayout()
        layout.addWidget(self.numLabel)
        layout.addWidget(self.numTubeSel)
        layout.addWidget(self.dataLabel)
        layout.addWidget(self.browse)
        layout.addWidget(self.source)
        layout.addStretch(1)
        self.topLeftGroupBox.setLayout(layout)

    def numTube(self):
        self.nTube = self.numTubeSel.value()
        # print('Number of tubes: ', self.numTube)
        return self.numTube

    def browseClick(self):
        self.folder = str(QFileDialog.getExistingDirectory(None, "Select Folder"))
        self.source.setText(self.folder)
        self.src = str(self.source.text())

        return self.src

    def createTopRightGroupBox(self):
        self.topRightGroupBox = QGroupBox("Display images")

        self.allImg = QCheckBox("All")
        self.originalImg = QCheckBox("Original")
        self.thresholdImg = QCheckBox("Threhsold")
        self.markedImg = QCheckBox("Marked")

        self.allImg.toggled.connect(self.originalImg.setChecked)
        self.allImg.toggled.connect(self.thresholdImg.setChecked)
        self.allImg.toggled.connect(self.markedImg.setChecked)

        self.originalImg.toggled.connect(self.displayImg)
        self.thresholdImg.toggled.connect(self.displayImg)
        self.markedImg.toggled.connect(self.displayImg)

        layout = QVBoxLayout()
        layout.addWidget(self.allImg)
        layout.addWidget(self.originalImg)
        layout.addWidget(self.thresholdImg)
        layout.addWidget(self.markedImg)
        layout.addStretch(1)
        self.topRightGroupBox.setLayout(layout)

    def displayImg(self):
        self.original = False
        self.threshold = False
        self.marked = False

        if self.originalImg.isChecked():
            self.original = True

        if self.thresholdImg.isChecked():
            self.threshold = True

        if self.markedImg.isChecked():
            self.marked = True

        return self.original, self.threshold, self.marked

    def createBottomLeftTabWidget(self):
        self.bottomLeftTabWidget = QTextEdit()

        self.bottomLeftTabWidget.setPlainText("Distance from origin\n"
                              "\n"
                              "J1:\n"
                              "\n"
                              "J2:\n"
                              "\n"
                              "Tip:\n")

    def createBottomRightGroupBox(self):
        self.bottomRightGroupBox = QGroupBox("Start/Stop")

        self.startBtn = QPushButton("Start")
        self.startBtn.resize(100,100)
        # self.startBtn.setCheckable(True)
        # self.startBtn.setChecked(False)
        self.isStarted = False
        self.startBtn.pressed.connect(self.start)

        self.stopBtn = QPushButton("Stop")
        self.isStopped = True
        self.stopBtn.pressed.connect(self.stop)

        layout = QGridLayout()
        layout.addWidget(self.startBtn)
        layout.addWidget(self.stopBtn)

        self.bottomRightGroupBox.setLayout(layout)

    def start(self):
        self.isStarted = True
        self.isStopped = False
        os.system("python 3dgui.py 1")

        return self.isStarted

    def stop(self):
        self.isStarted = False
        self.isStopped = True

if __name__ == '__main__':

    app = QApplication(sys.argv)
    gui = CTR_GUI()
    # gui.show()

    ntube = gui.nTube
    oImg, thImg, mImg = gui.displayImg()
    src = gui.source.text()
    print(src)
    st = gui.isStarted

    sys.exit(app.exec_())
