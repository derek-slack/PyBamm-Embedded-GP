import csv
import os

in_directory = "/Users/derekslack/Pybamm-Embedded-GP-live/src/Data/RawData/MACCOR Data-2"
# in_directory = "/Users/derekslack/Pybamm-Embedded-GP-live/src/Data/RawData/Re_ MACCOR Data"
out_directory = "/Users/derekslack/Pybamm-Embedded-GP-live/src/Data/Processed_Data"
for root,dirs,files in os.walk(in_directory):
    for file in files:
       if file.endswith(".csv"):
        file_in = os.path.join(in_directory,file)
        file_out = os.path.join(out_directory,file)
        with open(file_in, newline="") as f_in, open(file_out, "w", newline="") as f_out:
            reader = csv.reader(f_in)
            writer = csv.writer(f_out)

            meta_data = next(reader)
            meta_date = next(reader)
            header = next(reader)
            ncols = len(header)-1
            writer.writerow(header)

            for row in reader:
                if len(row) > ncols:
                    # merge the split "1,000" back into "1000"
                    row = [row[0] + row[1]] + row[2:]
                writer.writerow(row)