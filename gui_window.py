import sys
import math
import time
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from neural_network import neuralNetwork
from readCSV import revertPredict

class mainWindow(QMainWindow, QWidget):
    def __init__(self,parent = None):
        super(mainWindow,self).__init__(parent)
                
        self.form_widget = formWidget(self)
        self.setCentralWidget(self.form_widget)

        # set menu bar of main window
        menubar = self.menuBar()
        menubar = self.create_Menus(menubar)        
        
        # set toolbar menu
        toolbar = self.addToolBar("toolbar")
        toolbar = self.create_Toolbars(toolbar)

        # set statusBar
        self.statusBar = QStatusBar()
        #self.statusBar.showMessage("Status bar")
        self.setStatusBar(self.statusBar)

        #set location and size of main window        
        self.setGeometry(100,50,700,500) # (x,y) -> position on monitor, (width, height) -> dimension of window
        #self.setBaseSize(100,900)
        self.setWindowTitle("Automatic Landmarking on Beetle's Anatomicals")
            

    def create_Menus(self,menuBar):
        filemn = menuBar.addMenu("File")
        #editmn = menuBar.addMenu("Edit")
        #viewmn = menuBar.addMenu("View")
        helpmn = menuBar.addMenu("Help")
    
        openImg_act = QAction("Open image", self)
        openImg_act.setShortcut("Ctrl+I")
        openImg_act.triggered.connect(self.menu_openImg_Clicked)
        filemn.addAction(openImg_act)

        openModel_act = QAction("Open model", self)
        openModel_act.setShortcut("Ctrl+M")
        openModel_act.triggered.connect(self.menu_openModel_Clicked)
        filemn.addAction(openModel_act)

        filemn.addSeparator()

        saveact = QAction("Save",self)
        saveact.setShortcut("Ctrl+S")
        saveact.triggered.connect(self.menu_Save_Clicked)
        filemn.addAction(saveact)
    
        exitact = QAction("Exit",self)
        exitact.triggered.connect(exit)
        filemn.addAction(exitact)

        run_act = QAction("How to run?",self)
        run_act.triggered.connect(self.menu_how_to_run_Clicked)
        helpmn.addAction(run_act)

        helpmn.addSeparator()

        about_act = QAction("About",self)
        about_act.triggered.connect(self.menu_aboutact_Clicked)
        helpmn.addAction(about_act)

        return menuBar
    
    def create_Toolbars(self,toolBar):
        openImage = QAction(QIcon("images/photos.png"), "Open image", self)
        toolBar.addAction(openImage)
        
        openModel = QAction(QIcon("images/model.png"), "Open model", self)
        toolBar.addAction(openModel)
        
        saveAct = QAction(QIcon("images/save.svg"), "Save", self)
        toolBar.addAction(saveAct)

        toolBar.actionTriggered[QAction].connect(self.toolbar_Clicked)
        return toolBar
    
    # Menu events
    def menu_openImg_Clicked(self):
        self.form_widget.btnImage_Clicked()

    def menu_openModel_Clicked(self):
        self.form_widget.btnModel_Clicked()

    def menu_Save_Clicked(self):
        self.form_widget.saveMenu_Clicked()

    def menu_aboutact_Clicked(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("About")
        msg_box.setWindowTitle("Landmarks CNN Prediction")
        msg_box.setText("Landmarks CNN Prediction is designed by MorphoBoid group.\n"
                        + "This is a tool to test the CNN models on beetle's anatomical images. \n"
                        + "More information: http://morphoboid.labri.fr")
        msg_box.exec_()
    
    def menu_how_to_run_Clicked(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("How to run program")
        msg_box.setText("<b>The steps to use program:</b>")
        msg_box.setInformativeText("1. Select the test image\n"
            + "2. Select the trained model \n"
            + "3. Click on \"Run the test\" button"
        )
        msg_box.exec_()
    
    # Toolbar events
    def toolbar_Clicked(self,toolAct):
        cLabel = toolAct.text()
        if cLabel == "Open image":
            self.menu_openImg_Clicked()
        if cLabel == "Open model":
            self.menu_openModel_Clicked()
        if cLabel == "Save":
            self.menu_Save_Clicked()

DEFAULT_IMAGE = 'data/pronotum.JPG'
DEFAULT_MODEL = 'data/m_pronotum_fn_v10.pickle'

class formWidget(QWidget):
    
    def __init__(self,parent = None):
        super(formWidget,self).__init__(parent)
        self.create_Splitters()
        self.image = '' #data/Prono_001.JPG'
        self.model = '' #data/cnnmodel_8_output_96x96_10000epochs_change_rate_01001_change_kernel_3x32x22x2_v10.pickle'    
        self.pimage = None
        self.current_layer_id = -1
        self.outputs = []           
    

    def create_Splitters(self):
        
        hbox = QHBoxLayout(self)
        
        topleft = self.create_TopLeft()
        topright = self.create_TopRight()
     
        splitter1 = QSplitter(Qt.Horizontal)
        
        splitter1.addWidget(topleft)        
        splitter1.addWidget(topright)
        splitter1.setSizes([300,400])
        
        hbox.addWidget(splitter1)

        self.setLayout(hbox)

    def create_TopLeft(self):

        topleft = QWidget()
        layout = QVBoxLayout(self)
        
        self.btnImage = QPushButton('Load Image')
        self.btnImage.setIcon(QIcon("images/photos.png"))
        #self.btnImage.setFixedSize(fixedSize)
        #self.btnImage.setIconSize(fixedSize)
        self.btnImage.clicked.connect(self.btnImage_Clicked)

        self.limage = QLabel("Image")
        self.limage.setAlignment(Qt.AlignCenter)
        self.limage.setFrameShape(QFrame.Panel)
        
        self.btnModel = QPushButton('Load trained model')
        self.btnModel.setIcon(QIcon("images/model.png"))
        #btnModel.setFixedSize(fixedSize)
        #btnModel.setIconSize(fixedSize)
        self.btnModel.clicked.connect(self.btnModel_Clicked)

        self.lmodel = QLabel("Model")
        self.lmodel.setAlignment(Qt.AlignCenter)
        self.lmodel.setFrameShape(QFrame.Panel)

        self.btnRun = QPushButton('Run a test')
        self.btnRun.clicked.connect(self.btnRun_Clicked)

        layout.addWidget(self.btnImage)
        layout.addWidget(self.limage)
        layout.addWidget(self.btnModel)
        layout.addWidget(self.lmodel)
        layout.addWidget(self.btnRun)
        topleft.setLayout(layout)
        return topleft


    def create_TopRight(self):
        topright = QWidget()
        layout = QVBoxLayout(self)
        
        self.glLabel = QLabel('')
        self.glLabel.setAlignment(Qt.AlignCenter)
        self.glLabel.setStyleSheet('background-color: white;font-weight:bold; color: red; font-size: 16px')
        #self.glLabel.setSizeIncrement(256,192)
        layout.addWidget(self.glLabel)

        self.textEdit = QTextEdit()
        self.textEdit.setMaximumHeight(200)
        layout.addWidget(self.textEdit)
        
        topright.setLayout(layout)
        return topright


    def btnImage_Clicked(self):
        d = QFileDialog()
        fname = d.getOpenFileName(self,"Open image", "data","Image files (*.JPG)")
        if fname == '':
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Load data")
            msg_box.setText("No image has been selected. \n Would you like to use the default image?")
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            rtval = msg_box.exec_()
            if rtval == QMessageBox.Yes:
                fname = DEFAULT_IMAGE
        if fname != '':
            self.image = str(fname)
            self.pimage = QPixmap(fname)
            self.pimage = self.pimage.scaled(256,192)
            self.limage.setPixmap(self.pimage)
        
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Load data")
            msg_box.setText("Image has been loaded!")
            msg_box.exec_()
            self.btnImage.setText('Change image')

    def btnModel_Clicked(self):
        d = QFileDialog()
        fname = d.getOpenFileName(self,"Open model", "data","Pickle files (*.pkl | *.pickle)")
        if fname == '':
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Load data")
            msg_box.setText("No model has been selected. \n Would you like to use the default model?")
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            rtval = msg_box.exec_()
            if rtval == QMessageBox.Yes:
                fname = DEFAULT_MODEL
        if fname != '':
            self.model = str(fname)
            pixmap = QPixmap("images/model.png")
            pixmap = pixmap.scaled(256,256)
            self.lmodel.setPixmap(pixmap)  
             
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Load data")
            msg_box.setText("Model have been loaded!")
            msg_box.exec_()
            self.btnModel.setText('Change model')

    def btnRun_Clicked(self):
        if self.image == '':
            msg_box_img = QMessageBox()
            msg_box.setWindowTitle("Warning")
            msg_box_img.setIcon(QMessageBox.Question)
            msg_box_img.setText('The test image had not been loaded.\n Would you like to use the default image?')
            msg_box_img.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            img_retval = msg_box_img.exec_()
            if img_retval == QMessageBox.Yes:
                self.image = DEFAULT_IMAGE
                self.pimage = QPixmap(self.image)
                self.limage.setPixmap(self.pimage)
        if self.model == '':
            msg_box_model = QMessageBox()
            msg_box.setWindowTitle("Warning")
            msg_box_model.setText('The model had not been loaded.\n Would you like to use the default model?')
            msg_box_model.setIcon(QMessageBox.Question)
            msg_box_model.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            model_retval = msg_box_model.exec_()
            if model_retval == QMessageBox.Yes: 
                self.model = DEFAULT_MODEL
                modelpx = QPixmap("images/model.png")
                modelpx = modelpx.scaled(256,256)
                self.lmodel.setPixmap(modelpx)

        msg_box = QMessageBox()
        msg_box.setWindowTitle("Finish")
        if self.image != '' and self.model != '':            
            cnn = neuralNetwork(self.image, self.model)
            self.y_preds = cnn.predict()            
             
            real_lm = revertPredict(self.y_preds)
            self.print_prediction(real_lm)
            self.glLabel.setPixmap(self.pimage)
                    
            msg_box.setText("Finish process !")            
        else:
            msg_box = QMessageBox()
            msg_box.setText("The image and the model do not empty !")
        msg_box.exec_()

    def print_prediction(self,preds):        
        x, y = preds[:,::2], preds[:,1::2]
        qp = QPainter(self.pimage)
        qp.setPen(QColor(Qt.red))
        qp.setFont(QFont('Arial',13))

        self.textEdit.append("Number of landmarks:" + str(x.shape[1]) +"\n")
        for i in range(x.shape[1]):
            xi = x[0,i]
            yi = y[0,i]
            qp.fillRect(xi,yi,4,4,Qt.red)
            self.textEdit.append(str(xi) + " " + str(yi) + "\n")

    def saveMenu_Clicked(self,):
        d = QFileDialog()
        fname = d.getSaveFileName(self,"Open image", "results","Image files (*.JPG)")
        if fname =='':
            return
        else:
            save_file = QFile(fname)
            save_file.open(QIODevice.WriteOnly)
            if self.pimage is not None:
                self.pimage.save(save_file,"JPG")
            return

