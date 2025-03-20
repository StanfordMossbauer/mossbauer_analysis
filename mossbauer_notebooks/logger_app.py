from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QListWidget, QGridLayout, QLabel, \
    QFileDialog, QLineEdit
from PyQt5.QtCore import QTimer, QDateTime
import pyqtgraph as pg
import numpy as np

# import nidaqmx
from os import path
import csv

from yoctopuce.yocto_api import *
from yoctopuce.yocto_temperature import *
from yoctopuce.yocto_humidity import *

from datetime import datetime

# filenames for logs
filename='templog092021'
filename2='humilog092021'
# save to a G-drive folder
path_name = r'G:\Shared drives\RAY\data\Yoctopuce'
path_name = '.'
filename = path_name + '/' +filename
filename2 = path_name + '/' +filename2

class loggerWidget(QWidget):
    """A widget that plots and logs sensor data. Must add to main window.
    Parameters:
            name (str): Measured quantity; humidity, temperature, e.g.
            sensor_method (func): Function that returns current sensor measurement.
    """
    def __init__(self, name, filename, sensor_method, parent=None):
        super(loggerWidget, self).__init__(parent)
        layout = QGridLayout()
        self.setLayout(layout)

        self.read = sensor_method

        # file name and dir
        # self.log_file = os.getcwd() + "/" + filename + ".csv"
        self.log_file = filename + ".csv"

        self.title_widget = QLabel(name)
        self.vals = [self.read()]
        self.times = [datetime.now()]

        self.paused = False

        # Use a timer to update plot and log data
        self.timer = QTimer()

        # Default sampling rate = 1 minute
        self.defaultTime = 1
        self.timer.start(int(self.defaultTime *1000* 60))

        # Trigger update function on timeout
        self.timer.timeout.connect(self.update)



        # INITIALIZE THE PLOT WIDGET
        self.plot = pg.PlotWidget()
        # Change pen color + thickness
        pen = pg.mkPen(color=(114, 166, 151), width=3)
        # White background
        self.plot.setBackground('w')

        # Turn the x-axis into a date-time axis
        axis = pg.DateAxisItem()
        self.plot.setAxisItems({'bottom': axis})

        # This is the plotted data/"curve"
        self.curve = self.plot.getPlotItem().plot(pen=pen)

        #INITIALIZE THE LABELS / BUTTON WIDGETS

        # Display the log file
        self.save_txt = QLabel('Saving to ' + self.log_file)
        self.save_txt.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        # In-line editor to change the sampling time
        self.sample_label = QLabel("Sampling interval (min)")
        self.sample_time = QLineEdit(str(self.defaultTime), self)
        # Click to focus (display the blinking cursor)
        self.sample_time.setFocusPolicy(QtCore.Qt.ClickFocus)

        # Only accept numbers > .001 (minutes)
        self.sample_time.setValidator(QDoubleValidator(bottom=.001))
        # Upon finished, trigger function to change sample time
        self.sample_time.editingFinished.connect(self.new_sample_time)
        self.sample_time.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)

        # Change log file button
        self.file_btn = QPushButton("Change file")
        self.file_btn.clicked.connect(self.get_file)

        # Pause logging button
        self.pause_btn = QPushButton("Pause logger")
        self.pause_btn.setCheckable(True)
        self.pause_btn.clicked.connect(self.toggle_pause)

        # Add the widgets to the layout
        row1 = QGridLayout()
        row2 = QGridLayout()
        row2.addWidget(self.title_widget, 1, 1)
        row2.addWidget(self.sample_label, 1, 2)
        row2.addWidget(self.sample_time, 1, 3)
        row2.addWidget(self.pause_btn, 1, 4)

        row1.addWidget(self.save_txt, 1, 1)
        row1.addWidget(self.file_btn, 1, 2)
        layout.addLayout(row1, 1,1)
        layout.addLayout(row2, 2,1)
        layout.addWidget(self.plot, 3,1)

    def new_sample_time(self):
        """Modify the timer's timeout value. """
        # If the in-line editor value has changed:
        if self.sample_time.isModified():
            # Fetch the text from the in-line editor.
            time = int(float(self.sample_time.text()) * 1000 * 60)
            # Restart the timer.
            self.timer.stop()
            self.timer.start(time)
            # Reset the modified flag.
            self.sample_time.setModified(False)

    def get_file(self):
        fname = QFileDialog.getSaveFileName(self, 'Save file',
                                            'c:\\', "CSV files (*.csv)")
        if fname[0] != "":
            self.log_file = fname[0]
        #Modify the file name widget.
        self.save_txt.setText('Saving to ' + self.log_file)

    def toggle_pause(self):
        if self.pause_btn.isChecked():
            self.paused = True
        else:
            self.paused = False

    def update(self):
        """Updates the plot and logs new data."""
        if not self.paused:
            value = self.read()
            time = datetime.now()
            self.vals = np.append(self.vals, value)
            self.times.append(time)
            # Must convert the datetime object to timestamp format
            time_axis = [i.timestamp() for i in self.times]
            self.curve.setData(time_axis, self.vals)
            timeDisplay = QDateTime.currentDateTime().toString('yyyy-MM-dd hh:mm:ss')
            # Save to file. Append if exists, otherwise create.
            if path.exists(self.log_file):
                type = 'a'
            else:
                type = 'w'
            with open(self.log_file, type, newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timeDisplay, value])


class WinForm(QWidget):
    def __init__(self, parent=None):
        super(WinForm, self).__init__(parent)
        self.setWindowTitle('Logger')

        # Arrange the widgets on a grid
        layout = QGridLayout()
        self.setLayout(layout)

        # Timestamp widget
        self.time_label = QLabel()
        self.date_timer = QTimer()
        self.date_timer.start(1000)
        self.date_timer.timeout.connect(self.updateTime)

        # Status widgets
        self.daq_txt = QLabel()
        self.temp_txt = QLabel()
        self.hum_txt = QLabel()

        # Arrange the time and status texts.
        row0 = QGridLayout()
        row0.addWidget(self.time_label, 1, 1)
        row0.addWidget(self.temp_txt, 1, 2)
        row0.addWidget(self.hum_txt, 1, 3)
        row0.addWidget(self.daq_txt, 1, 4)
        layout.addLayout(row0, 1, 1)

        # Set up temperature logger
        errmsg = YRefParam()
        #if YAPI.RegisterHub("127.0.0.1", errmsg) != YAPI.SUCCESS:
        #    sys.exit("init error :" + errmsg.value)
        
        if YAPI.RegisterHub("usb", errmsg) != YAPI.SUCCESS:
            print('fail')
        try:
            self.temperature = YTemperature.FirstTemperature()
            self.temps = [self.temperature.get_currentValue()]
            self.tempFound = True
            self.tempstatus = ""
        except Exception as e:
            self.tempFound = False
            print(e)
            self.tempstatus = "Temp. sensor error."

        try:
            self.humidity = YHumidity.FirstHumidity()
            self.hums = [self.humidity.get_currentValue()]
            self.humFound = True
            self.humstatus = ""
        except Exception as e:
            self.humFound= False
            print(e)
            self.humstatus = "Humidity sensor error."

        # If sensors found, create plot panels.
        if self.tempFound:
            temp_panel= loggerWidget("Temperature", filename, self.temperature.get_currentValue)
            layout.addWidget(temp_panel, 2,1)
            temp_panel.show()

        if self.humFound:
            hum_panel = loggerWidget("Humidity", filename2, self.humidity.get_currentValue)
            layout.addWidget(hum_panel, 2, 2)
            hum_panel.show()

        # Check if NI DAQ working/connected
        # try:
        #     nidaqmx.Task()
        #     daqstatus = ""
        #     self.daqFound = True
        # except FileNotFoundError:
        #     self.daqFound = False
        #     daqstatus = "DAQ not found."

        # If DAQ found, create plot panel.
        # if self.daqFound:
        #     dloggaq_panel = loggerWidget("Lock status", "lock_log", self.temperature.get_currentValue)
        #     layout.addWidget(daq_panel, 3,1)
        #     daq_panel.show()

        # Update the status texts.
        self.hum_txt.setText(self.humstatus)
        self.temp_txt.setText(self.tempstatus)
        #self.daq_txt.setText(daqstatus)

    # Function to update the timestamp.
    def updateTime(self):
        timeDisplay = QDateTime.currentDateTime().toString('yyyy-MM-dd hh:mm:ss')
        self.time_label.setText(timeDisplay)


app = QApplication(sys.argv)
form = WinForm()
form.show()
sys.exit(app.exec_())
