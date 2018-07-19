import re
import os
from os import listdir
from os import walk
from os.path import isfile, join
from datetime import datetime
import datetime as dt

class Scan():
    def __init__(self, upc, datetime, timestamp):
        self.upc = upc
        self.timestamp = timestamp
        self.datetime = datetime

class Receipt():
    def __init__(self):
        self.prices = []
        self.items = []
        self.items_count = []
        self.datetime = None
    def get_total(self):
        total = 0
        for price in self.prices:
            total+= price
        return total
    def __repr__(self):
        return "Items: {0}\nQTY: {1}\nPrices: {2}\nTotal: {3}$\nTime of purchase: {4}\nUPCs: {5}\n-----------".format(self.items, self.items_count, self.prices, self.total, self.datetime, self.upcs)

def process_file(receipt_path, scans_path):
    receipt_lines = open(receipt_path,'r')
    text = receipt_lines.readlines()
    receipt =  extract_receipt_data(text)
    items_count = len(receipt.items)

    #find most probable upcs for the scans
    upcs = find_relevant_upcs(scans_path, items_count, receipt.datetime)
    receipt.upcs = upcs
    return receipt
#Receipt comprehension trough regex matching
def extract_receipt_data(lines):
    receipt = Receipt()
    previous = ""
    try:
        for line in lines:
           
            #find line which contains SKU
            sku = re.findall('(\d+)\s+sku', line)
            if sku:
                items_bought = int(sku[0])
                
                parsed_prices = []
                new_prices = re.findall("\-*\d+\.\d+", line)
                if new_prices:
                    for price in new_prices:
                        parsed_price = float(price)
                        parsed_prices.append(parsed_price)
                
                receipt.prices.append(parsed_prices[-1])
                receipt.items.append(previous)
                receipt.items_count.append(items_bought)

            #find exact date time in the specific for BevMax format
            date = re.findall('\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} [AP][M]', line)
            date_found = next(iter(date), None)
            if(date_found):
                prased_datetime = datetime.strptime(date_found, "%m/%d/%Y %I:%M:%S %p")
                #print(prased_datetime)
            if date:
                receipt.datetime = prased_datetime
            #TODO: Chck for subtotal matching
            total_price = re.findall('TOTAL\s+(\d+.\d+)', line)
            if total_price:
                #print(total_price[0] + " TOTAL")
                receipt.total = float(total_price[0])

            previous = line.rstrip('\n')

        return receipt
    except:
        return None
    
def find_relevant_upcs(folder, items_count, target_datetime):
    dates = []
    items_upcs = []
    timestamps = []
    #Traverse data files
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    for (dirpath, dirnames, filenames) in walk(folder):
        for filename in filenames:
            try:
                selected_datetime = timestamp_to_datetime(filename)
                #Skip irrelevante newer date
                if target_datetime < selected_datetime:
                    #print("skipped {0} target {1}".format(selected_datetime, target_datetime))
                    continue
                
                timestamps.append(filename)
                dates.append(selected_datetime)
                file = open(folder + "/" + filename, 'r') 
                text = file.read().rstrip('\x00')
                decoded = bytearray.fromhex(text).decode()
                items_upcs.append(decoded)
            except:
                #print("Unable to decode")
                pass
    #copy store uuid n times
    store_UUID = os.path.basename(folder) 
    
    #sort data by date
    sorted_lists = sorted(zip(dates, items_upcs, timestamps), reverse=False, key=lambda x: x[0])
    dates, items_upcs, timestamps = [[x[i] for x in sorted_lists] for i in range(3)]
    print(timestamps[:items_count])
    return items_upcs[:items_count]

#project time stamp to datetime
def timestamp_to_datetime(timestamp):
    s = int(timestamp) / 1000.0
    time = dt.datetime.fromtimestamp(s)
    return time
    #datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S')

print(process_file('data/SamplePrintScans/bevmax_sample_receipt_01.txt', '/home/quelibrio/Work/Datasets/SampleScans'))
print(process_file('data/SamplePrintScans/bevmax_sample_receipt_02.txt', '/home/quelibrio/Work/Datasets/SampleScans'))
print(process_file('data/SamplePrintScans/bevmax_sample_receipt_03.txt', '/home/quelibrio/Work/Datasets/SampleScans'))
print(process_file('data/SamplePrintScans/bevmo_sample_receipt_01.txt', '/home/quelibrio/Work/Datasets/SampleScans'))

#receipt2 = process_file('data/SamplePrintScans/imaginary_files.txt')
#print(receipt2.datetime, receipt2.items)