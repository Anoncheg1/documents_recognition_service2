import os
import requests
import time

url = "http://127.0.0.1:5000"


def pdf_processing(input_file):
    files = {'pdf': open(input_file, 'rb')}
    job_id = requests.post(url + "/upload", files=files).json()["id"]
    print("PDF FILE: " + input_file)
    print("JOB ID: " + str(job_id))
    response = ""
    outr = ""

    previous_response = "in pool"
    while response != "ready":
        outr = requests.get(url + "/get?id=" + job_id).json()
        response = outr["status"]
        if response != previous_response:
            previous_response = response
            print("*" + response + "*")
        time.sleep(10)

    dirtyflag = True
    for item in outr["pages"]:
        if item["qc"] == 2 or item["qc"] == 3:
            dirtyflag = False
        elif len(item["period"]) == 0:
            dirtyflag = False

    if dirtyflag == True:
        os.rename(input_file, "./good/" + input_file)
        print(input_file + " -> оке")
    else:
        os.rename(input_file, "./bad/" + input_file)
        print(input_file + " -> проблемы")
    print("--------------------------")


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
