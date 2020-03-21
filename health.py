from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql.types import *
from pyspark.sql import SQLContext
import os
import pathlib

conf = SparkConf()
conf.setMaster('local')
conf.setAppName('test')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

def mod(x):
    import numpy as np
    return (x, np.mod(x, 2))
rdd = sc.parallelize(range(1000)).map(mod).take(10)
print(rdd)


patient_directories = [(f.path, f.name) for f in os.scandir("./CheXpert-v1.0-small/train") if f.is_dir()]


def handle_patient_dir(patient_dir):
    (patient_path, patient_name) = patient_dir
    study_directories = [(f.path, f.name) for f in os.scandir(patient_path) if f.is_dir()]
    for study_path, study_name in study_directories:
        frontal_images = [(f.path, f.name) for f in os.scandir(study_path) if f.is_file() and "frontal" in f.name.lower()]
        for image_path, image_name in frontal_images:
            temp_path = study_path + "/temp" + image_name
            return ",".join([patient_name, patient_path, study_name, study_path, image_name, image_path, temp_path])



patient_directories_rdd = sc.parallelize(patient_directories)
csv_rows = patient_directories_rdd.map( handle_patient_dir )
csv_rows.saveAsTextFile("./frontal_images.csv")
