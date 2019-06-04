from MLP import MLP_Model
from load_data import Data
from ui import *
from sklearn.model_selection import train_test_split
import sys

class Main(Ui_MainWindow):
    def __init__(self,MainWindow):
        super().setupUi(MainWindow)
        self.doubleSpinBox.setRange(0.0,1.0)
        self.doubleSpinBox.setValue(0.3)
        self.spinBox.setMinimum(1)
        self.spinBox_2.setRange(20,10001)
        self.doubleSpinBox_2.setRange(0,1.0)
        self.doubleSpinBox_2.setValue(0.5)
        self.mode=None
        self.connect()

    def connect(self):
        self.pushButton.clicked.connect(self.init)
        self.pushButton_2.clicked.connect(self.train)
        self.pushButton_3.clicked.connect(self.predict)
        self.pushButton_4.clicked.connect(self.show)

    def init(self):
        # data
        dataset = self.comboBox.currentText()
        data_process = self.comboBox_2.currentText()
        test_size = float(self.doubleSpinBox.value())
        data, label = Data().load(dataset,data_process)
        if data is None:
            self.textBrowser.setText("[Error]: input error" + str(label))
            return
        self.train_data, self.test_data, self.train_label, self.test_label = train_test_split(data, label,test_size=test_size)
        n_inputs = len(data[0])
        self.label = list(set(i for i in label))
        self.n_outputs = len(self.label)
        n_hidden_layer = int(self.spinBox.value())
        hidden_layer = self.lineEdit.text().split()
        if len(hidden_layer) == 0:
            self.textBrowser.setText("[Error]: input error")
            return
        try:
            hidden_layer = list(map(int, hidden_layer[:n_hidden_layer]))
        except Exception as e:
            self.textBrowser.setText("[Error]: input error, " + str(e))
            return

        self.MLP = MLP_Model(n_inputs,hidden_layer,self.label)
        self.textBrowser.setText("[Success]: init sucesss")
        self.mode ='init'

    def train(self):
        if self.mode==None :
            self.textBrowser.setText("[Error]: it does not init the network")
            return
        n_epoch=int(self.spinBox_2.value())
        learning_rate=float(self.doubleSpinBox_2.value())
        self.textBrowser.setText("[Waiting]: training.......")
        self.activation = self.comboBox_3.currentText()
        status,text=self.MLP.train(n_epoch,self.train_data,self.train_label,learning_rate,self.activation)
        if status:
            self.textBrowser.setText("[Success]: training successs")
            self.textBrowser_2.setText(text)
        else:
            self.textBrowser.setText("[Error]: training error" + text)
        self.mode='train'

    def predict(self):
        # if self.mode!='train':
        #     self.textBrowser.setText("[Error]: it does not train the network")
        #     return
        self.textBrowser.setText("[Waiting]: predicting.......")
        predictions=self.MLP.predict(self.test_data)

        print(self.test_label)
        print(predictions)
        confusion_matrix = [[0 for i in range(self.n_outputs)] for j in range(self.n_outputs)]
        for i,j in zip(self.test_label,predictions):
            confusion_matrix[self.label.index(i)][self.label.index(j)]+=1

        accuracy=0
        for i in range(self.n_outputs):
            accuracy+=confusion_matrix[i][i]
        accuracy_rate=accuracy / len(predictions)

        self.textBrowser.setText("[predict]: accuracy rate: %lf" % (accuracy_rate))
        self.textBrowser_2.append("there is %d sample(s) in the test set\naccuracy: %d  ,accuracy rate: %lf\n"%(len(predictions),accuracy,accuracy_rate))
        for i,j in zip(self.label,confusion_matrix):
            self.textBrowser_2.append(str(i)+"   "+str(j)+'\n')

        self.textBrowser_2.append("actual:\n"+str(self.test_label))
        self.textBrowser_2.append("predictions:\n" + str(predictions))


    def show(self):
        if self.mode!='train':
            self.textBrowser.setText("[Error]: it does not train the network")
            return
        self.MLP.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui =Main(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())