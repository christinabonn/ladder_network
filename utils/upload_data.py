import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import io
import requests
from zipfile import ZipFile


class DataUploader(object):

  @staticmethod
  def upload_and_read_from_url(url, read_file_name, save_file_name):
    # Check if data was downloaded, otherwise download it and save for future use
    save_file_name = os.path.join('../data', save_file_name)
    if os.path.isfile(save_file_name):
      text_data = []
      with open(save_file_name, 'r') as temp_output_file:
        reader = csv.reader(temp_output_file)
        for row in reader:
          text_data.append(row)
    else:
      zip_url = url
      r = requests.get(zip_url)
      z = ZipFile(io.BytesIO(r.content))
      file = z.read(read_file_name)
      # Format Data
      text_data = file.decode()
      text_data = text_data.encode('ascii',errors='ignore')
      text_data = text_data.decode().split('\n')
      text_data = [x.split('\t') for x in text_data if len(x)>=1]
      
      # And write to csv
      with open(save_file_name, 'w+') as temp_output_file:
          writer = csv.writer(temp_output_file)
          writer.writerows(text_data)

    return text_data