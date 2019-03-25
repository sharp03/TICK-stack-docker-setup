import time
import pyodbc
from influxdb import InfluxDBClient
import datetime
import pandas as pd
from pandas import DataFrame
import os
data_base = "mydb"
w_measurement = "sqldata_new"
ctr = 0

book = xlrd.open_workbook("data.txt")
sheet1 = book.sheet_by_name("Sheet1")

for r in range(1,sheet1.nrows):
        indate                  = sheet1.cell(r,0).value
        value1                  = sheet1.cell(r,1).value
        time                    = int((float(indate)-25569) * 86400)
        print(ctr,time)
        os.system("docker-compose exec 'http://localhost:8086/write?db=%s&precision=s' --data-binary '%s value=%d %d'" % (data_base,w_measurement,value1,time))
        ctr += 1
