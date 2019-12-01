#!/usr/bin/env python
import datetime, sys, numpy as np , pandas as pd , matplotlib.pyplot as plt,matplotlib.dates as mdates
from PyQt5.QtCore import QDateTime, Qt, QTimer , QDate , QUrl , QFileInfo
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget , QDateEdit , QCalendarWidget , QListWidget ,
        QFileDialog, QMessageBox, QMainWindow, QTableView)
from PyQt5.QtWebEngineWidgets import (QWebEngineView , QWebEnginePage,
         QWebEngineSettings)
from PyQt5.QtNetwork import *
from PyQt5.QtCore import pyqtSlot

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
import matplotlib
matplotlib.use("Qt5Agg")

from PandasModel import PandasModel

from scipy import interpolate
import seaborn as sns
sns.set(rc={'figure.figsize':(11,4)})
from pyramid.arima import auto_arima


df_sales = pd.read_excel(io="Sales Data.xls",header=1)
df_production = pd.read_excel(io="Production Data.xls",header=1,index_col='Mill Name')

df_sales["Mill Name"]=df_sales["Mill Name"].replace({'A':'Burhanpur Tapti Textile Mills',
                             'B':'Aarti Cotton Textile Mills',
                             'C':'Barshi Textile Mills',
                            'D':'Finlay Mill Achalpur',
                             'E':'Finlay Mill Mumbai',
                             'F':'India United Mills No5',
                            'G':'New Bhopal Textile Mills',
                             'H':'Podar Mills',
                             'I':'Rajnagar Textile Mills',
                            'J':'Tata Mills'})


df_sales_enc=df_sales.assign(date_cat = lambda row :
                       row["Enquiry Date"].str.split(' ').str.get(0))
df_sales_enc.drop(labels=[ 'Enquiry No' , 'Order No' , 'Enquiry Date',
                      'Status','Discount Value','GST Value',
                     'Net Ex-Mill Invoice Value',
                     'Total Invoice Value' , 'Ex-Mill Invoice Value','Party Type'] , axis=1,inplace=True)


df_sales_enc['Date']=pd.to_datetime(df_sales_enc.loc[:,"date_cat"],dayfirst=True)
df_sales_enc.set_index("Date" , inplace=True)
df_sales_enc.sort_index(inplace=True)
df_sales_enc["weighted_rate"] = df_sales_enc["Total Bags"]*df_sales_enc["Rate"]
df_sales.reset_index(inplace=True)
df_sales_enc["date_cat"] = df_sales_enc.index

df_prod_enc=df_production.dropna(how='any')
df_prod_enc["Date"]=pd.to_datetime(df_prod_enc["Date"],dayfirst=True)
df_prod_enc.reset_index(inplace=True)
df_prod_enc.set_index("Date" , inplace=True)
df_prod_enc["date_cat"] = df_prod_enc.index


count_wise=pd.read_excel('Count wise sale from Nov 15 to feb 19 (1).xlsx')
count_wise.columns = count_wise.iloc[3]


count_wise=count_wise.reindex(count_wise.index.drop([0,1,2,3]))
count_wise['Count'].fillna(method='ffill',inplace=True)
count_wise=count_wise.fillna(0)


count_wise=count_wise.reset_index().drop(['index'],axis=1)
count_wise=count_wise.set_index(['Count'])
count_wise=count_wise.drop([datetime.datetime(2019, 3, 1, 0, 0),'Total'],axis=1)
complete=count_wise[count_wise != 0].dropna()

RTM_40k=complete.iloc[0,:].drop(['Mill Name'])
BT_54=complete.iloc[1,:].drop(['Mill Name'])
Tata_60P=complete.iloc[2,:].drop(['Mill Name'])
Podar_60P=complete.iloc[3,:].drop(['Mill Name'])
Indu_62P=complete.iloc[4,:].drop(['Mill Name'])

VAR={"MILL_NAME":["All",'Burhanpur Tapti Textile Mills', 'Podar Mills',
                'Rajnagar Textile Mills', 'India United Mills No5',
                'Finlay Mill Mumbai', 'Tata Mills', 'Barshi Textile Mills',
                'Aarti Cotton Textile Mills', 'Finlay Mill Achalpur',
                'New Bhopal Textile Mills'] , 
       "ITEM_NAME":["All",'Cotton yarn',
                'Hank yarn',
                'Polyester-cotton',
                'Polyester yarn',
                'Polyester-viscose'],
        "VARIETY":['All','60s POLYSTER 100%',
                '56s PC 70:30 AUTO',
                '40s KD AUTO',
                '60s KD COMPACT AUTO',
                '62s POLYSTER 100% EYC',
                '50s Poly 100% AUTO HT TPI 38',
                '45s Poly 100% AUTO HT TPI 35',
                '40s KD COMPACT 100 AUTO',
                '34s KD AUTO',
                '54s PC 70:30 AUTO',
                '40s KD CONE HOSIERY AUTO',
                '50s KD COMPACT 100 AUTO',
                '60s PC 67:33 AUTO',
                '60s Poly 100% AUTO HT TPI 38',
                '65s POLYSTER 100%',
                '2/60s PC 67:33',
                '40s PC 70:30 AUTO',
                '30s PC 67:33',
                '67s CBD COMPACT AUTO',
                '60s KD CONE AUTO',
                '40s PV 65:35 AUTO',
                '60s PC 67:33',
                '40s KD COMPACT AUTO',
                '62s POLYSTER AUTO',
                '46s K AUTO Cone',
                '29s K AUTO Cone',
                '46s KD AUTO'], 
        "Graph":['line' ,'box', 'bar','area' , 'hist', 'scatter'],
        "PRED_MILL_NAME":{"RTM_40k":RTM_40k,"BT_54":BT_54,
                "Tata_60P":Tata_60P,"Podar_60P":Podar_60P,"Indu_62P":Indu_62P}}
                

def BEST(m,train,valid,date_range):
        train=train
        valid=valid
        future=date_range
        model = auto_arima(train, start_p=24, start_q=24,start_Q=12,start_P=12,max_P=48,max_Q=48,max_p=48, max_q=48,m=m, max_order=96,seasonal=True,d=1, D=1, trace=False,error_action='ignore',suppress_warnings=True)
        model.fit(train)

        forecast = model.predict(n_periods=len(valid)+len(date_range))
        forecast = pd.DataFrame(forecast,index = np.append(valid.index ,date_range) ,columns=['Prediction'])
        return forecast

class Root(QMainWindow):
        def __init__(self, parent=None):
        # """INIT ALL THE METHODS HERE"""        
                super().__init__(parent)
                self.setWindowTitle("NTC Sales")
                self.init_data() #VARIABLES 
                self.init_Layout()
                self.main_layout_init()


        def init_data(self):
        # """INIT ALL VARIABLES HERE"""
                self.mill_name = 'All'
                self.date1 = '2018-01-09'
                temp= QDate.currentDate().toString("yyyy-MM-dd")
                self.date2 = temp
                self.pred_date = '2019-06-01'
                self.item_name = 'All'
                self.variety = 'All'
                self.graph_type = 'line'
                self.browser = Browser()
                self.url = QUrl("https://www.ntcltd.org/Home.aspx")
                self.fig , self.axes = plt.subplots(3,1,figsize=(15,30))
                self.pred_mill_name = BT_54
                self.A = pd.date_range('2019-03-01', '2019-06-01', freq='MS')
                self.mul_variety = []
                self.and_mul_var = 1 

        def init_Layout(self):
        # """INIT LAYOUT"""
                self.mainLayout = QGridLayout()
                self.topLayout = QGridLayout()
                self.topGroup = QGroupBox("Mill Properties")
                self.middleLayout = QGridLayout()
                self.middleGroup = QGroupBox()
                self.bottomLayout = QGridLayout()
                self.bottomGroup = QGroupBox()


        def main_layout_init(self):
        # """MAIN LAYOUT HERE""" 
                self.top_layout_init()
                self.middle_layout_init()
                self.bottom_layout_init()

                self.mainLayout.addWidget(self.topGroup, 0, 0, 1, 2)
                self.mainLayout.addWidget(self.middleGroup, 1, 0, 1, 2)
                self.mainLayout.addWidget(self.bottomGroup , 2,0,1,2)
                self.mainLayout.setRowStretch(0, 0)
                self.mainLayout.setRowStretch(1, 0)
                self.mainLayout.setColumnStretch(0, 1)
                self.mainLayout.setColumnStretch(1, 1)
                tWidget = QWidget()
                tWidget.setLayout(self.mainLayout)
                self.setCentralWidget(tWidget)

        def top_layout_init(self):
        # """TOP LAYOUT"""
                firstLayer = QHBoxLayout()
                secondLayer = QHBoxLayout()
                thirdLayer = QHBoxLayout()

                millComboBox = QComboBox()
                millComboBox.addItems(VAR["MILL_NAME"])
                millLabel = QLabel("&Mill Name:")
                millLabel.setBuddy(millComboBox)
                millComboBox.activated[str].connect(self.set_mill)

                date_1_label = QLabel("&Date:")
                date_1 = QDateEdit()
                date_1.setDate(QDate(2018 , 1 , 1))
                date_1.setMinimumDate(QDate(2018 , 1 , 1))
                date_1.setCalendarPopup(True)
                date_1_label.setBuddy(date_1)
                date_1.dateChanged.connect(self.set_date1)

                date_2 = QDateEdit()
                date_2.setDate(QDate.currentDate())
                date_2.setMinimumDate(QDate(2018 , 1 , 1))
                date_2.setCalendarPopup(True)
                date_2.dateChanged.connect(self.set_date2)

                itemComboBox = QComboBox()
                itemComboBox.addItems(VAR["ITEM_NAME"])
                itemLabel = QLabel("&Item Name:")
                itemLabel.setBuddy(itemComboBox)
                itemComboBox.activated[str].connect(self.set_item)
                
                varietyComboBox = QComboBox()
                varietyComboBox.addItems(VAR["VARIETY"])
                varietyLabel = QLabel("&Variety:")
                varietyLabel.setBuddy(varietyComboBox)
                varietyComboBox.activated[str].connect(self.set_variety)

                self.varSpecific = QLineEdit()
                self.varSpecific.setToolTip("Give comma seperated fields of what you want")

                varAndOr = QCheckBox()
                varAndOr.setChecked(True)
                varAndOr.toggled.connect(self.set_var_and_or)

                predMill = QComboBox()
                predMill.addItems(list(VAR["PRED_MILL_NAME"].keys()))
                predLabel = QLabel("&Mill_for_pred:")
                predLabel.setBuddy(predMill)
                predMill.activated[str].connect(self.set_pred_mill)

                predDate_label = QLabel("&Pred_Date:")
                predDate = QDateEdit()
                predDate.setDate(QDate(2019 , 6 , 1))
                predDate.setMinimumDate(QDate(2019 , 5 , 1))
                predDate.setCalendarPopup(True)
                predDate_label.setBuddy(predDate)
                predDate.dateChanged.connect(self.set_pred_date)



                firstLayer.addWidget(millLabel)
                firstLayer.addWidget(millComboBox)
                firstLayer.addStretch(10)
                firstLayer.addWidget(date_1_label)
                firstLayer.addWidget(date_1)
                firstLayer.addWidget(QLabel("-"))
                firstLayer.addWidget(date_2)

                secondLayer.addWidget(itemLabel)
                secondLayer.addWidget(itemComboBox)
                # secondLayer.addStretch(5)
                secondLayer.addWidget(varietyLabel)
                secondLayer.addWidget(varietyComboBox)
                secondLayer.addWidget(QLabel("&Mul Var:"))
                secondLayer.addWidget(self.varSpecific)
                secondLayer.addWidget(QLabel("&contain_all"))
                secondLayer.addWidget(varAndOr)



                thirdLayer.addWidget(predLabel)
                thirdLayer.addWidget(predMill)
                thirdLayer.addStretch(5)
                thirdLayer.addWidget(predDate_label)
                thirdLayer.addWidget(predDate)
                
                self.topLayout.addLayout(firstLayer , 0 ,0 , 1 , 2)
                self.topLayout.addLayout(secondLayer , 1 ,0 , 1,2)
                self.topLayout.addLayout(thirdLayer , 2 ,0 , 1 , 2)

                self.topGroup.setLayout(self.topLayout)

        
        def middle_layout_init(self):
        # Set Bottom Layout
                graphCombobox = QComboBox()
                graphCombobox.addItems(VAR["Graph"])
                graphLabel = QLabel("&Graph Type")
                graphLabel.setBuddy(graphLabel)
                graphCombobox.activated[str].connect(self.set_graph)

                predictButton =QPushButton()
                predictButton.setText("&Get Prediction")
                predictButton.clicked.connect(self.get_prediction)
                
                graphButton =QPushButton()
                graphButton.setText("&Get Graph")
                graphButton.clicked.connect(self.get_graph)

                rawButton = QPushButton()
                rawButton.setText("&Get Raw")
                rawButton.clicked.connect(self.get_raw)
                
                middleLayer = QHBoxLayout()
                middleLayer.addWidget(graphLabel)
                middleLayer.addWidget(graphCombobox)
                # middleLayer.addStretch(1)
                middleLayer.addWidget(graphButton)
                
                bottomlayer = QHBoxLayout()
                bottomlayer.addWidget(predictButton)
                bottomlayer.addWidget(rawButton)
                
                self.middleLayout.addLayout(bottomlayer,1,0,1,3)
                self.middleLayout.addLayout(middleLayer,0,0,1,3)
                
                self.middleGroup.setLayout(self.middleLayout)

        def bottom_layout_init(self):
                webButton = QPushButton()
                webButton.setText("&Go To Site")
                webButton.clicked.connect(self.get_web)

                quitButton = QPushButton()
                quitButton.setText("Quit")
                quitButton.clicked.connect(self.close)

                self.bottomLayout.addWidget(webButton , 0, 0)
                self.bottomLayout.addWidget(quitButton , 0, 1)
                self.bottomGroup.setLayout(self.bottomLayout)


        def set_pred_mill(self , mn):
                self.pred_mill_name = VAR["PRED_MILL_NAME"][mn]
                print("Setting Pred Mill To "+self.pred_mill_name)
                print(self.pred_mill_name)

        def set_pred_date(self, d):
                self.pred_date = d.toString("yyyy-MM-dd")
                self.A=pd.date_range('2019-03-01', str(self.pred_date) , freq='MS')
                print("Setting Prediction Date to:",end=" ")
                print(self.pred_date)

        def set_mill(self , mill_name):
                self.mill_name = mill_name       
                print("Setting Mill NAme to:",end=" ")
                print(self.mill_name)    

        def set_var_and_or(self):
                self.and_mul_var = not(self.and_mul_var)
                print("Setting Multiple variety To OR:", end=" ")
                print(self.and_mul_var)
        
        def set_item(self , item_name):
                self.item_name = item_name.upper()
                print("Setting Item:",end=" ")
                print(self.item_name)

        def set_date1(self , date1):
                self.date1=date1.toString("yyyy-MM-dd")
                print("Setting Date1:" , end=" ")
                print(self.date1)

        def set_date2(self , date2):
                self.date2=date2.toString("yyyy-MM-dd")
                print("Setting date2:", end=" ")
                print(self.date2)

        def set_variety(self,variety):
                self.variety = variety
                print("Setting variety to: "+self.variety)

        def set_graph(self,graph):
                self.graph_type = graph
                print("Setting Graph to: "+self.graph_type)

        def get_raw(self):
                self.set_df_params()
                self.rawData = RawDataWindow(self._df_sales_enc_r , self._df_prod_enc_r)
                self.rawData.show()  

        def get_graph(self):
                self.set_df_params()
                self.fig , self.axes = plt.subplots(3,1,figsize=(15,30))
                self.set_sales_graph()
                self.set_production_graph()
                self.get_graph_window()
                print("SUCCESS!")

        def get_prediction(self):
                print("in prediction...")
                forecast = self.train_mill(self.pred_mill_name)
                self.predWindow= GraphWindow(self.fig_pred)
                self.predWindow.show()

                self.forWid = QWidget()
                self.forView = QTableView()
                model_forecast = PandasModel(forecast.reset_index())
                self.forView.setModel(model_forecast)
                self.forView.show()
                print("prediction complete...")


        def train_mill(self , mil):   
                bes_m=0
                best_rms=10000
                best_for=0
                print("IN TRAIN_MILL:")
                print(self.A)
                for i in range(2,25):
                        train = mil[:36]
                        valid = mil[36:]
                        forc=BEST(i,train,valid,self.A)
                        rms=np.sqrt(np.mean(np.power((np.array(valid)-np.array(forc['Prediction'][:len(valid)])),2)))
                        if rms<best_rms:
                                best_for=forc
                                bes_m=i
                                best_rms=rms
                print((str)(bes_m)+" "+(str)(best_rms)+"\t"+str(best_for))        
                self.fig_pred,self.ax_pred = plt.subplots(figsize=(12,12))
                self.ax_pred=plt.plot(train,label="Total Invoice")
                self.ax_pred=plt.plot(valid , label="Actual Value")
                self.ax_pred=plt.plot(best_for['Prediction'] , label="Prediction")
                plt.ylabel("Total InVoice", fontsize=15)
                plt.legend()
                # plt.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
                return best_for

        def set_df_params(self):
        #Setting the Parameters Set
        # Can make it to more options if more Combobox added
        # To Do: Make options for graphs to be plotted. Make a combox and give the variables here        
                self.mul_variety = self.varSpecific.text().split(",") #It takes multiple varities to be plotted

                if self.mill_name != "All":
                        temp_a1_sales = (df_sales_enc["Mill Name"] == self.mill_name)
                        temp_a1_prod = df_prod_enc["Mill Name"] == self.mill_name
                else:
                        temp_a1_sales = np.ones((len(df_sales_enc["Mill Name"])),dtype=bool)
                        temp_a1_prod = np.ones((len(df_prod_enc["Mill Name"])),dtype=bool)


                if self.item_name.capitalize() != "All":
                        temp_a3_prod = df_prod_enc["Item Name"] == self.item_name
                else:
                        temp_a3_prod = np.ones((len(df_prod_enc["Item Name"])),dtype=bool)


                if self.variety != "All":
                        temp_a2_sales = df_sales_enc["Variety Name"] == self.variety
                        temp_a2_prod = df_prod_enc["Variety Name"] == self.variety

                elif len(self.mul_variety)!=0:
                        temp_a2_sales = np.zeros((len(df_sales_enc["Variety Name"])),dtype=bool)
                        temp_a2_prod = np.zeros((len(df_prod_enc["Variety Name"])),dtype=bool)
                else:
                        temp_a2_sales = np.ones((len(df_sales_enc["Variety Name"])),dtype=bool)
                        temp_a2_prod = np.ones((len(df_prod_enc["Mill Name"])),dtype=bool)


                if not(self.and_mul_var):
                        temp_mul_var1 = np.zeros((len(df_sales_enc["Variety Name"])),dtype=bool)
                        temp_mul_var1_prod = np.zeros((len(df_prod_enc["Variety Name"])),dtype=bool)

                elif len(self.mul_variety) == 0:
                        temp_mul_var1 = np.zeros((len(df_sales_enc["Variety Name"])),dtype=bool)
                        temp_mul_var1_prod = np.zeros((len(df_prod_enc["Variety Name"])),dtype=bool)     
                else:
                        temp_mul_var1 = np.ones((len(df_sales_enc["Variety Name"])),dtype=bool)
                        temp_mul_var1_prod = np.ones((len(df_prod_enc["Variety Name"])),dtype=bool)
                        

                #Sets the multi variable params in this for loop
                for mul_var_str in self.mul_variety:
                        if len(self.mul_variety) == 0:
                                break
                        if not(self.and_mul_var):
                                temp_mul_var1 = np.logical_or(temp_mul_var1 , df_sales_enc["Variety Name"].str.contains(str(".*"+mul_var_str+".*"),case=False))
                                temp_mul_var1_prod = np.logical_or(temp_mul_var1_prod , df_prod_enc["Variety Name"].str.contains(str(".*"+mul_var_str+".*"),case=False))
                        else:
                                temp_mul_var1 = np.logical_and(temp_mul_var1 , df_sales_enc["Variety Name"].str.contains(str(".*"+mul_var_str+".*"),case=False))
                                temp_mul_var1_prod = np.logical_and(temp_mul_var1_prod , df_prod_enc["Variety Name"].str.contains(str(".*"+mul_var_str+".*"),case=False))
                        
                        
                temp_a2_sales = np.logical_or(temp_a2_sales , temp_mul_var1)
                temp_a2_prod = np.logical_or(temp_a2_prod , temp_mul_var1_prod)

                try:

                        self._df_sales_enc_g = df_sales_enc[np.logical_and(temp_a1_sales , temp_a2_sales)]
                        self._df_sales_enc_g = self._df_sales_enc_g.loc[self.date1:self.date2]
                        self._df_sales_enc_r = self._df_sales_enc_g.copy()
                        
                        self._df_sales_enc_g=self._df_sales_enc_g.groupby(level=0).sum()
                        self._df_sales_enc_g["weighted_rate"] = self._df_sales_enc_g["weighted_rate"]/self._df_sales_enc_g["Total Bags"]
                        self._df_sales_enc_g_7roll = self._df_sales_enc_g.rolling(7,center=True).mean()
                        self._df_sales_enc_g_30roll = self._df_sales_enc_g.rolling(30,center=True).mean()
                        
                        self._df_sales_enc_g["Year"] = self._df_sales_enc_g.index.year
                        self._df_sales_enc_g["Month"] = self._df_sales_enc_g.index.month
                        self._df_sales_enc_g["Weekday Name"] = self._df_sales_enc_g.index.weekday_name


                        self._df_prod_enc_g = df_prod_enc[np.logical_and(temp_a1_prod,np.logical_and(temp_a2_prod,temp_a3_prod))]
                        self._df_prod_enc_g = self._df_prod_enc_g.loc[self.date1 : self.date2]
                        self._df_prod_enc_r = self._df_prod_enc_g.copy()

                        self._df_prod_enc_g=self._df_prod_enc_g.groupby(level=0).sum()
                        self._df_prod_enc_g.where(self._df_prod_enc_g["Weights"]<80000 ,self._df_prod_enc_g["Weights"].median(),inplace=True)
                        # self._df_prod_enc_g.where(self._df_prod_enc_g["Weights"]>15000 ,self._df_prod_enc_g["Weights"].median(),inplace=True)
                        
                        self._df_prod_enc_g_7roll = self._df_prod_enc_g.rolling(7,center=True).mean()
                        self._df_prod_enc_g_30roll = self._df_prod_enc_g.rolling(30,center=True).mean()
                        
                        self._df_prod_enc_g["Year"] = self._df_prod_enc_g.index.year
                        self._df_prod_enc_g["Month"] = self._df_prod_enc_g.index.month
                        self._df_prod_enc_g["Weekday Name"] = self._df_prod_enc_g.index.weekday_name
                        print("Success in setting params!")
                except Exception as e:
                        print("Exception in setting df_params")
                        print(e)

        def set_sales_graph(self):
        #Set self.axes[0] , self.axes[1] of self.fig to plot in FigureCanvas

                try:
                        if self.graph_type == "line":
                                self.axes[0].plot(self._df_sales_enc_g["weighted_rate"],marker=".",linestyle='-',linewidth=0.5,label='Daily Rate')
                                self.axes[0].plot(self._df_sales_enc_g_7roll["weighted_rate"],linestyle='-',color="red",label="Trend")
                                self.axes[0].plot(self._df_sales_enc_g_30roll["weighted_rate"],linestyle='-',color="black",label="Trend")
                                self.axes[1].plot(self._df_sales_enc_g["Total Bags"],marker=".",linestyle='-',linewidth=0.5,label='Daily Total Bags')
                                self.axes[1].plot(self._df_sales_enc_g_7roll["Total Bags"],linestyle='-',color="red",label="Trend")
                                self.axes[1].plot(self._df_sales_enc_g_30roll["Total Bags"],linestyle='-',color="black",label="Trend")
                                for ax in range(2):
                                        self.axes[ax].legend()
                                        # self.axes[ax].xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
                                        self.axes[ax].set_xlabel("Date - >",fontsize=10)
                        elif self.graph_type == "bar":
                                self.fig , self.axes = plt.subplots(5,2,figsize=(15,10))
                                sns.barplot(data = self._df_sales_enc_g , x="Weekday Name" , y="weighted_rate",ax=self.axes[0,0])
                                sns.barplot(data = self._df_sales_enc_g , x="Weekday Name" , y="Total Bags", ax=self.axes[0,1])
                                sns.barplot(data = self._df_sales_enc_g , x="Month" , y="weighted_rate", ax=self.axes[1,0])
                                sns.barplot(data = self._df_sales_enc_g , x="Month" , y="Total Bags", ax=self.axes[1,1])
                                sns.barplot(data = self._df_sales_enc_g , x="Year" , y="weighted_rate", ax=self.axes[2,0])
                                sns.barplot(data = self._df_sales_enc_g , x="Year" , y="Total Bags", ax=self.axes[2,1])
                        elif self.graph_type == "box":
                                self.fig , self.axes = plt.subplots(5,2,figsize=(15,10))
                                sns.boxplot(data = self._df_sales_enc_g , x="Weekday Name" , y="weighted_rate",ax=self.axes[0,0],showfliers=False)
                                sns.boxplot(data = self._df_sales_enc_g , x="Weekday Name" , y="Total Bags", ax=self.axes[0,1],showfliers=False)
                                sns.boxplot(data = self._df_sales_enc_g , x="Month" , y="weighted_rate", ax=self.axes[1,0],showfliers=False)
                                sns.boxplot(data = self._df_sales_enc_g , x="Month" , y="Total Bags", ax=self.axes[1,1],showfliers=False)
                                sns.boxplot(data = self._df_sales_enc_g , x="Year" , y="weighted_rate", ax=self.axes[2,0],showfliers=False)
                                sns.boxplot(data = self._df_sales_enc_g , x="Year" , y="Total Bags", ax=self.axes[2,1],showfliers=False)
                        elif self.graph_type == "scatter":
                                self.axes[0].plot(self._df_sales_enc_g_7roll["weighted_rate"],marker="o",linestyle='None',linewidth=0.5,label='Weighted Rate')
                                self.axes[1].plot(self._df_sales_enc_g_7roll["Total Bags"],marker="o",linestyle='None',linewidth=0.5 , label="Total Bags",color="orange")
                                for ax in range(2):
                                        self.axes[ax].legend()
                                        # self.axes[ax].xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
                                
                        else:
                                self._df_sales_enc_g.plot( y="weighted_rate" , kind=self.graph_type , ax=self.axes[0],label="Weighted Rate")
                                self._df_sales_enc_g.plot( y="Total Bags" , kind=self.graph_type , ax=self.axes[1], label="Total Bags")
                                
                except Exception as e:
                        print("Sales Exception:")
                        print(e)
                        QMessageBox.question(self , "","No Sales data" , QMessageBox.Ok,QMessageBox.Ok)

        def set_production_graph(self):
        #Set the self.axes[2] of self.fig (plt.figure) to plot  
                try:
                        if self.graph_type == "line":
                                self.axes[2].plot(self._df_prod_enc_g["Weights"],marker=".",linestyle='-',linewidth=0.5,label='Daily Weights')
                                self.axes[2].plot(self._df_prod_enc_g_7roll["Weights"],linestyle='-',color="red",label="Trend")
                                self.axes[2].plot(self._df_prod_enc_g_30roll["Weights"],linestyle='-',color="black",label="Trend")
                                self.axes[2].legend()
                        elif self.graph_type == "bar":
                                sns.barplot(data = self._df_prod_enc_g , x="Weekday Name" , y="Weights", ax=self.axes[3,0])
                                sns.barplot(data = self._df_prod_enc_g , x="Month" , y="Weights", ax=self.axes[3,1])
                                sns.barplot(data = self._df_prod_enc_g , x="Year" , y="Weights", ax=self.axes[4,0])
                        elif self.graph_type == "box":
                                sns.boxplot(data = self._df_prod_enc_g , showfliers=False, x="Weekday Name" , y="Weights", ax=self.axes[3,0])
                                sns.boxplot(data = self._df_prod_enc_g , showfliers=False, x="Month" , y="Weights", ax=self.axes[3,1])
                                sns.boxplot(data = self._df_prod_enc_g , showfliers=False, x="Year" , y="Weights", ax=self.axes[4,0])
                        elif self.graph_type == "scatter":
                                self.axes[2].plot(self._df_prod_enc_g_7roll["Weights"],marker="o",linestyle='None',linewidth=0.5 , label="Weights",color="orange")
                                self.axes[2].legend()
                                
                        else:
                                self._df_prod_enc_g.plot( y="Weights" , kind=self.graph_type , ax=self.axes[2], label="Weights")
                                
                except Exception as e:
                        print("Production Graph exception:")
                        print(e)
                        QMessageBox.question(self , "","No Production data" , QMessageBox.Ok,QMessageBox.Ok)
        
        def get_graph_window(self):
                self.fig.subplots_adjust(top=0.95,
                        bottom=0.095,
                        left=0.095,
                        right=0.95,
                        hspace=0.21,
                        wspace=0.2)
                self.graphWindow=GraphWindow(self.fig)
                self.graphWindow.show()

        def get_web(self):
                self.browser.load(self.url)
                self.browser.showMaximized()

class MyBrowser(QWebEnginePage):
#     ''' Settings for the browser.'''
    
        def userAgentForUrl(self, url):
                # ''' Returns a User Agent that will be seen by the website. '''
                return "Mozilla/5.0"

class Browser(QWebEngineView):
        def __init__(self):
                # QWebEngineView
                self.view = QWebEngineView.__init__(self)
                #self.view.setPage(MyBrowser())
                self.setWindowTitle('Loading...')
                self.titleChanged.connect(self.adjustTitle)
                self.settings = QWebEngineSettings.globalSettings()
                self.settings.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls , True)
                self.settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls,True)
                self.settings.setAttribute(QWebEngineSettings.JavascriptEnabled,True)
                self.settings.setAttribute(QWebEngineSettings.PluginsEnabled,True)
                self.settings.setAttribute(QWebEngineSettings.JavascriptCanOpenWindows , True)
                self.page().profile().downloadRequested.connect(self._downloadRequested)
        
                #super(Browser).connect(self.ui.webView,QtCore.SIGNAL("titleChanged (const QString&amp;)"), self.adjustTitle)

        def load(self,url):
                self.setUrl(QUrl(url))
        
        def adjustTitle(self):
                self.setWindowTitle(self.title())
        
        def disableJS(self):
                settings = QWebEngineSettings.globalSettings()
                settings.setAttribute(QWebEngineSettings.JavascriptEnabled, False)

        def createWindow(self, windowType):
                if windowType == QWebEnginePage.WebBrowserTab:
                        self.webView = Browser()
                        self.webView.show()
                        return self.webView
        
        def _downloadRequested(self,download): # QWebEngineDownloadItem
                old_path = download.url().path()  # download.path()
                suffix = QFileInfo(old_path).suffix()
                path, _ = QFileDialog.getSaveFileName(
                self, "Save File", old_path, "*." + suffix
                )
                if path:
                        download.setPath(path)
                        download.accept()

class PlotCanvas(FigureCanvas):
        # '''Plotting the Graph on a window. Just give the plt.
        # i/p=fig to be plotted
        # parent=parent of the canvas'''
        
    def __init__(self, fig , parent=None, width=5, height=4, dpi=100):
        
        super().__init__(fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.draw()

class GraphWindow(QWidget):
        # '''Pops up a Graph Window
        # Give the plt.fig to be plotted and do .show() method which will pop up the GraphWindow with toolbar'''

        def __init__(self , fig):
                super(GraphWindow , self).__init__()
                self.graph_layer(fig)

        def graph_layer(self , fig):
                self.canvas = PlotCanvas(fig , self)
                self.navi_tool = NavigationToolbar(self.canvas , self)
                self.layout = QVBoxLayout()
                self.layout.addWidget(self.canvas)
                self.layout.addWidget(self.navi_tool)
                self.setGeometry(10,10,600,400)
                self.setWindowTitle("Graph")
                self.setLayout(self.layout)

class RawDataWindow(QTabWidget):
        # """To show the raw data window"""
        def __init__(self, sales_df , prod_df):
                super(RawDataWindow , self).__init__()
                self.setSizePolicy(QSizePolicy.Preferred , QSizePolicy.Ignored)
                self.setWindowTitle("Raw Data")
                
                #Two parent tabs salesTab and prodTab
                #Followed by two child Tabs in each
                self.salesTabP = QTabWidget()
                self.prodTabP =QTabWidget()

                self.salesTabC1 = QTableView()
                self.salesTabC1.setSortingEnabled(True)
                model_sales = PandasModel(sales_df)
                self.salesTabC1.setModel(model_sales)

                self.salesTabC2 = QTableView()
                self.salesTabC2.setModel(PandasModel(sales_df.describe()))
                
                self.salesTabP.addTab(self.salesTabC1 , "&Sales_Data")
                self.salesTabP.addTab(self.salesTabC2 , "&Sales_Summary")

                self.prodTabC1 = QTableView()
                self.prodTabC1.setSortingEnabled(True)
                model_prod = PandasModel(prod_df)
                self.prodTabC1.setModel(model_prod)

                self.prodTabC2 = QTableView()
                self.prodTabC2.setModel(PandasModel(prod_df.describe()))
                
                self.prodTabP.addTab(self.prodTabC1 , "&Production_Data")
                self.prodTabP.addTab(self.prodTabC2 , "&Production_Summary")

                self.addTab(self.salesTabP,"&Sales")
                self.addTab(self.prodTabP,"&Production")

                


if __name__=="__main__":
                
        appctxt = QApplication([])
        gallery = Root()
        gallery.show()
        sys.exit(appctxt.exec_())