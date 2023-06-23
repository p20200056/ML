import pandas as pd
import numpy as np
import os
import glob
import math
import csv
import time
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
        
        
def analyse_file(filename):
	#Reading the data file
	data = pd.read_csv(filename , names=range(8))
	#Filtering Sensor Data
	rotation_vector_data = data[data[7] == "Cywee Rotation Vector"]
	accelerometer_data = data[data[5] == "Cywee Accelerometer Sensor"]
	gyroscope_data = data[data[5] == "Cywee Gyroscope Sensor"]
	magnetometer_data = data[data[5] == "Cywee Magnetic field Sensor"]
	linear_acceleration_data = data[data[5] == "Cywee Linear Acceleration"]
	heart_rate_data = data[data[3] == "Cywee Heart Rate"]
	#Cleaning Data
	rotation_vector_data = rotation_vector_data.drop([1,7] , axis = 1)
	rotation_vector_data = rotation_vector_data.rename(columns = {0: "TimeStamp", 2: "Axis1", 3: "Axis2", 4: "Axis3", 5: "Axis4", 6: "Axis5"})
	rotation_vector_data = rotation_vector_data.astype(float)
	
	accelerometer_data=accelerometer_data.drop([1,5,6,7], axis = 1)
	accelerometer_data=accelerometer_data.rename(columns={0: "TimeStamp", 2: "AX", 3: "AY", 4: "AZ"})
	accelerometer_data = accelerometer_data.astype(float)
	ax = accelerometer_data["AX"] * accelerometer_data["AX"]
	ay = accelerometer_data["AY"] * accelerometer_data["AY"]
	az = accelerometer_data["AZ"] * accelerometer_data["AZ"]
	am = ax + ay + az
	am = am.apply(lambda x: math.sqrt(x))
	accelerometer_data["ARMS"] = am

	gyroscope_data=gyroscope_data.drop([1,5,6,7], axis = 1)
	gyroscope_data=gyroscope_data.rename(columns={0: "TimeStamp", 2: "GX", 3: "GY", 4: "GZ"})
	gyroscope_data = gyroscope_data.astype(float)
	gx = gyroscope_data["GX"] * gyroscope_data["GX"]
	gy = gyroscope_data["GY"] * gyroscope_data["GY"]
	gz = gyroscope_data["GZ"] * gyroscope_data["GZ"]
	gm = gx + gy + gz
	gm = gm.apply(lambda x: math.sqrt(x))
	gyroscope_data["GRMS"] = gm

	magnetometer_data=magnetometer_data.drop([1,5,6,7], axis = 1)
	magnetometer_data=magnetometer_data.rename(columns={0: "TimeStamp", 2: "MX", 3: "MY", 4: "MZ"})
	magnetometer_data = magnetometer_data.astype(float)
	mx = magnetometer_data["MX"] * magnetometer_data["MX"]
	my = magnetometer_data["MY"] * magnetometer_data["MY"]
	mz = magnetometer_data["MZ"] * magnetometer_data["MZ"]
	mm = mx + my + mz
	mm = mm.apply(lambda x: math.sqrt(x))
	magnetometer_data["MRMS"] = mm

	linear_acceleration_data=linear_acceleration_data.drop([1,5,6,7], axis = 1)
	linear_acceleration_data=linear_acceleration_data.rename(columns={0: "TimeStamp", 2: "ACCX", 3: "ACCY", 4: "ACCZ"})
	linear_acceleration_data=linear_acceleration_data.astype(float)

	heart_rate_data=heart_rate_data.drop([1,3,4,5,6,7], axis = 1)
	heart_rate_data=heart_rate_data.rename(columns={0: "TimeStamp", 2: "HR"})
	heart_rate_data=heart_rate_data.astype(float)
	
	#Add filtering code for all
	
	#Statistical analysis for all Data
	parameters = Statistical_Analysis(rotation_vector_data, accelerometer_data, gyroscope_data, magnetometer_data, linear_acceleration_data, heart_rate_data)
	
	#return all parameters calculated for file
	return parameters
	
	
	
def Statistical_Analysis(rotation_vector_data, accelerometer_data, gyroscope_data, magnetometer_data, linear_acceleration_data, heart_rate_data):

	parameters = []
	#Calculating Parameters for Cyclic vector
	for column in ["Axis1", "Axis2", "Axis3", "Axis4", "Axis5"]:
		parameters.extend(stat_analysis_column(rotation_vector_data[column]))
		
	#Calculating Parameters for Accelerometer
	for column in ["AX", "AY", "AZ", "ARMS"]:
		parameters.extend(stat_analysis_column(accelerometer_data[column]))
		
	#Calculating Parameters for Gyroscope
	for column in ["GX", "GY", "GZ", "GRMS"]:
		parameters.extend(stat_analysis_column(gyroscope_data[column]))
		
	#Calculating Parameters for Magnetometer
	for column in ["MX", "MY", "MZ", "MRMS"]:
		parameters.extend(stat_analysis_column(magnetometer_data[column]))
		
	#Calculating Parameters for Acceleration Vector
	for column in ["ACCX", "ACCY", "ACCZ"]:
		parameters.extend(stat_analysis_column(linear_acceleration_data[column]))
		
	#Calculating Parameters for Heart Rate
	for column in ["HR"]:
		parameters.extend(stat_analysis_column(heart_rate_data[column]))

	return parameters



def stat_analysis_column(data):
	features = [
	data.mean(skipna = True),
	data.std(skipna = True),
	data.var(skipna = True),
	data.min(skipna = True),
	data.max(skipna = True),
	data.skew(skipna = True),
	data.kurtosis(skipna = True)]
	#Add code for spectral entropy
	return features

data_source_path = "/Users/apoorva/Desktop/INSTRSOP/Fall-Detection-ML/TheOne/*"
data_destination_path = "/Users/apoorva/Desktop/INSTRSOP/Fall-Detection-ML/Features.csv"
i=0
l=389
nan_list = []
start=time.time()
Fall_occurance = 0
printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
with open(data_destination_path, 'w') as out_file:
	rows = csv.writer(out_file)
	
	list_of_folders = glob.glob(data_source_path)
	for folder in list_of_folders:
		if "FallData" in folder:
			Fall_occurance = 1
		else:
			Fall_occurance = 0
		
		list_of_folders_1 = glob.glob(folder + "/*")
		for sub_folder in list_of_folders_1:
			list_of_files = glob.glob(sub_folder + "/*.csv")
			for file in list_of_files:
				parameters = analyse_file(file)
				if True in pd.isna(parameters):
					nan_list.append(file)
				rows.writerow(parameters + [Fall_occurance])
				#print("Extraction Completed: " + file)
				i += 1
				printProgressBar(i, l, prefix = ' Progress:', suffix = 'Complete', length = 50)
with open("nan_files.txt", 'w') as file1:
        for i in nan_list:
            file1.write(i+'\n')
end=time.time()
print(end-start)
