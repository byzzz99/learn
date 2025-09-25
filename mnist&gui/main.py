import sys
from mygui import Ui_Form
from myfunction import myFunction
from PyQt5 import QtWidgets


if __name__=="__main__":
    app=QtWidgets.QApplication(sys.argv)
    mywindow=myFunction()
    mywindow.show()
    sys.exit(app.exec_())


