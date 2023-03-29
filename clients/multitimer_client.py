import os
import requests
import time

url = "http://127.0.0.1:5000"


def pdf_processing(input_file):
    for i in range(25):
        files = {'pdf': open(input_file, 'rb')}
        start_time = time.time()
        job_id = requests.post(url + "/upload", files=files).json()["id"]
        response = ""
        outr = ""

        previous_response = "in pool"
        while response != "ready":
            outr = requests.get(url + "/get?id=" + job_id).json()
            response = outr["status"]
            if response != previous_response:
                previous_response = response
            time.sleep(0.5)
        print(time.time() - start_time)


def main():
    thisfiles = []
    for root, dirs, files in os.walk("."):
        for filename in files:
            if filename[-3:] == "pdf":
                thisfiles.append(filename)

    for item in thisfiles:
        pdf_processing(item)


if __name__ == "__main__":
    main()
