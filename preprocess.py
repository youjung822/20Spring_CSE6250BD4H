from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SQLContext
import os
from PIL import Image

from resizeimage import resizeimage

conf = SparkConf()
conf.setMaster('local')
conf.setAppName('test')
sc = SparkContext(conf=conf)
log4jLogger = sc._jvm.org.apache.log4j
LOGGER = log4jLogger.LogManager.getLogger(__name__)
sqlContext = SQLContext(sc)


# walk through the directory and generate a list of frontal images in a csv with their path
def handle_patient_dir(patient_dir):
    (patient_path, patient_name) = patient_dir
    study_directories = [(f.path, f.name) for f in os.scandir(patient_path) if f.is_dir()]
    for study_path, study_name in study_directories:
        frontal_images = [(f.path, f.name) for f in os.scandir(study_path) if f.is_file() and "frontal" in f.name.lower()]
        for image_path, image_name in frontal_images:
            temp_path = study_path + "/temp" + image_name
            return ",".join([patient_name, patient_path, study_name, study_path, image_name, image_path, temp_path])


# Crop all frontal images in the RDD to be (320, 320)
def crop_all_images(df):
    rdd = df.rdd.map(tuple)

    def crop_image(row):
        old_path = row[5]
        new_path = row[6]
        if old_path is None:
            print("None old path")
            return
        with open(old_path, 'r+b') as f:
            with Image.open(f) as image:
                cover = resizeimage.resize_cover(image, [320, 320])
                cover.save(new_path, image.format)

    rdd.foreach(crop_image)


def main():

    patient_directories = [(f.path, f.name) for f in os.scandir("./CheXpert-v1.0-small/study") if f.is_dir()]
    patient_directories_rdd = sc.parallelize(patient_directories)
    csv_rows = patient_directories_rdd.map(handle_patient_dir)
    csv_rows.saveAsTextFile("./frontal_images.csv")
    image_crop_df = sqlContext.read.format('com.databricks.spark.csv').options(header='false', inferschema='true')\
        .load('./frontal_images.csv/part-00000')
    crop_all_images(image_crop_df)


if __name__== "__main__":
    main()
