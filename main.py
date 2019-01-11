import sys
import time
from PyQt4 import QtCore, QtGui
from gui_window import mainWindow
from PyQt4.QtGui import *

def main_window():
    app = QtGui.QApplication(sys.argv)
    ex = mainWindow()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main_window()
