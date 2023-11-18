import numpy as np
import math
import csv as commaSeparatedValues

# Inisiasi variabel-variabel
dataSource = "Alba"
baseDirectory = r'C:\Users\User\Documents\Project\indoor_localization\Raw' + '\\'
trainingRatio = 0.8
trainingLength = int(trainingRatio*200)
distanceOrder = 2
windowLength = 8
# Indexing tingkat kedua: 0 = -4 dBm, 1 = -12 dBm (tx)
rssi = np.empty([6, 2, 6, 7, 200], dtype=int)
# Indexing tingkat dua: tx. Indexing tingkat kelima: 0 = max, 1 = median, 2 = mean, 3 = varians
rssi_centering = np.empty([6, 2, 6, 7, 4], dtype=float)
# Indexing tingkat ketiga: centering
bayesianLikelihood = np.empty([6, 7, 3], dtype=float)
# Indexing tingkat keempet: centering. Indexing tingkat kelima: 0 = ordinat perkiraan, 1 = axis perkiraan, 2 = galat (meter)
prediction = np.empty([6, 7, 201-trainingLength-windowLength, 3, 3], dtype=float)
# Variabel untuk keluaran ke .csv
fields = ['Distribusi Kekuatan Transmisi', 'Tipe Pusatan Data', 'Rerata Galat (meter)', 'Standar Deviasi', 'ECDF 95% (meter)']
rows = np.empty(shape=(192,5), dtype='object')
counter = 0
# Definisi fungsi pembantu
def predict(tx):
    for axis in range(7):
        for ordinate in range(6):
            for test_number in range(201-trainingLength-windowLength):
                for calc_axis in range(7):
                    for calc_ordinate in range(6):
                        rp_data = np.empty([6], dtype=float)
                        test_data = np.empty([6], dtype=float)
                        distance_power = np.empty([3], dtype=float)
                        var = 0
                        # Max
                        for beacon in range(6):
                            rp_data[beacon] = rssi_centering[beacon, tx[beacon], calc_ordinate, calc_axis, 0]
                            test_data[beacon] = np.max(rssi[beacon, tx[beacon], ordinate, axis, trainingLength+test_number:trainingLength+windowLength+test_number])
                            var += rssi_centering[beacon, tx[beacon], calc_ordinate, calc_axis, 3]
                        var /= 6
                        distance_power[0] = np.sum(abs(rp_data-test_data)**distanceOrder)/6
                        # print("distance Mean : ", distance_power[0])
                          # Median
                        for beacon in range(6):
                            rp_data[beacon] = rssi_centering[beacon, tx[beacon], calc_ordinate, calc_axis, 1]
                            test_data[beacon] = np.median(rssi[beacon, tx[beacon], ordinate, axis, trainingLength+test_number:trainingLength+windowLength+test_number])
                        distance_power[1] = np.sum(abs(rp_data-test_data)**distanceOrder)/6
                        # print("distance Median : ", distance_power[1])
                        # Mean
                        for beacon in range(6):
                            rp_data[beacon] = rssi_centering[beacon, tx[beacon], calc_ordinate, calc_axis, 2]
                            test_data[beacon] = np.mean(rssi[beacon, tx[beacon], ordinate, axis, trainingLength+test_number:trainingLength+windowLength+test_number])
                            # if axis == 0 and calc_axis == 0 and calc_ordinate == 0 and test_number == 0 and beacon == 0:
                                # Print statements
                                # print(f"Beacon: {beacon}, Calc_Axis: {calc_axis}, Calc_Ordinate: {calc_ordinate}")
                                # print("RP data MEAN :", rp_data)
                                # print("test data MEAN : ", test_data)
                        distance_power[2] = np.sum(abs(rp_data-test_data)**distanceOrder)/6
                        # print("distance Max : ", distance_power[2])
                        # print('Distance : ', distance_power)
                        # print("RP DATA Beacon {} : ".format(beacon))
                        # print('TEST Data : ',test_data)
                        bayesianLikelihood[calc_ordinate, calc_axis, :] = np.exp(-0.5*distance_power**(2/distanceOrder)/var)
                        print("likelihood : ", bayesianLikelihood)
                        
                # Max
                coordinate = np.unravel_index(np.argmax(bayesianLikelihood[:, :, 0]), (6, 7))
                prediction[ordinate, axis, test_number, 0, 0] = coordinate[0]
                prediction[ordinate, axis, test_number, 0, 1] = coordinate[1]
                prediction[ordinate, axis, test_number, 0, 2] = math.sqrt((coordinate[0]-ordinate)**2 + (coordinate[1]-axis)**2)
                # Median
                coordinate = np.unravel_index(np.argmax(bayesianLikelihood[:, :, 1]), (6, 7))
                prediction[ordinate, axis, test_number, 1, 0] = coordinate[0]
                prediction[ordinate, axis, test_number, 1, 1] = coordinate[1]
                prediction[ordinate, axis, test_number, 1, 2] = math.sqrt((coordinate[0]-ordinate)**2 + (coordinate[1]-axis)**2)
                # Mean
                coordinate = np.unravel_index(np.argmax(bayesianLikelihood[:, :, 2]), (6, 7))
                prediction[ordinate, axis, test_number, 2, 0] = coordinate[0]
                prediction[ordinate, axis, test_number, 2, 1] = coordinate[1]
                prediction[ordinate, axis, test_number, 2, 2] = math.sqrt((coordinate[0]-ordinate)**2 + (coordinate[1]-axis)**2)
def tx_to_str(tx):
    if(tx == 0):
        return "-4"
    elif(tx == 1):
        return "-12"
    else:
        return "error"
# Membaca csv dan pemusatan data training
for axis in range(7):
    for ordinate in range(6):
        csv = np.genfromtxt(baseDirectory + dataSource + '\\' + str(axis) + ',' + str(ordinate) + "_" + dataSource + ".csv", delimiter=',', case_sensitive=False)
        for beacon in range(6):
            rssi[beacon, 0, ordinate, axis, :] = csv[beacon,5:205]
            rssi_centering[beacon, 0, ordinate, axis, 0] = np.max(rssi[beacon, 0, ordinate, axis, 0:trainingLength])
            rssi_centering[beacon, 0, ordinate, axis, 1] = np.median(rssi[beacon, 0, ordinate, axis, 0:trainingLength])
            rssi_centering[beacon, 0, ordinate, axis, 2] = np.mean(rssi[beacon, 0, ordinate, axis, 0:trainingLength])
            rssi_centering[beacon, 0, ordinate, axis, 3] = np.var(rssi[beacon, 0, ordinate, axis, 0:trainingLength])
        csv = np.genfromtxt(baseDirectory + dataSource + '-12\\' + str(axis) + ',' + str(ordinate) + "_" + dataSource + "-12.csv", delimiter=',', case_sensitive=False)
        for beacon in range(6):
            rssi[beacon, 1, ordinate, axis, :] = csv[beacon,5:205]
            rssi_centering[beacon, 1, ordinate, axis, 0] = np.max(rssi[beacon, 1, ordinate, axis, 0:trainingLength])
            rssi_centering[beacon, 1, ordinate, axis, 1] = np.median(rssi[beacon, 1, ordinate, axis, 0:trainingLength])
            rssi_centering[beacon, 1, ordinate, axis, 2] = np.mean(rssi[beacon, 1, ordinate, axis, 0:trainingLength])
            rssi_centering[beacon, 1, ordinate, axis, 3] = np.var(rssi[beacon, 1, ordinate, axis, 0:trainingLength])
# print("RSSI at point {0},{1}".format(axis, ordinate))
# print(rssi[beacon, 0, 0, 1, :])
tx_value = [1, 0, 1, 0, 1, 1, 0, 1]
predict(tx_value)
# Perkiraan lokasi dengan Bayesian Estimator sekaligus mengukur galat perkiraan
# for tx1 in range(2):
#     for tx2 in range(2):
#         for tx3 in range(2):
#             for tx4 in range(2):
#                 for tx5 in range(2):
#                     for tx6 in range(2):
#                         predict([tx1, tx2, tx3, tx4, tx5, tx6])
                        # for centeringType in range(3):
#                             print(counter)
#                             rows[counter, 0] = tx_to_str(tx1) + ", " + tx_to_str(tx2) + ", " + tx_to_str(tx3) + ", " + tx_to_str(tx4) + ", " + tx_to_str(tx5) + ", " + tx_to_str(tx6)
#                             if(centeringType == 0):
#                                 rows[counter, 1] = "Max"
#                             elif(centeringType == 1):
#                                 rows[counter, 1] = "Median"
#                             elif(centeringType == 2):
#                                 rows[counter, 1] = "Mean"
#                             else:
#                                 rows[counter, 1] = "Error"
#                             rows[counter, 2] = np.mean(prediction[:, :, :, centeringType, 2])
#                             rows[counter, 3] = np.std(prediction[:, :, :, centeringType, 2])
#                             rows[counter, 4] = np.percentile(prediction[:, :, :, centeringType, 2], 95)
#                             counter += 1
# with open(r'D:\Universitas_Gadjah_Mada\Akademis--TIF_2018\Capstone\Python\Spreadsheet\Bayesian Estimator Asimetri' + '\\' + dataSource + " 80-20 Power " + str(distanceOrder) + " Window " + str(windowLength) + ".csv", 'w', newline='') as csvFile:
#     csvWriter = commaSeparatedValues.writer(csvFile)
#     csvWriter.writerow(fields)
#     csvWriter.writerows(rows)