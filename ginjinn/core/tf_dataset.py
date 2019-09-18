import tensorflow as tf
import xml.etree.ElementTree as ET
import pandas as pd
import glob
import json
import os
import io

from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import namedtuple, OrderedDict
from tqdm import tqdm
from PIL import Image
from object_detection.utils import dataset_util, label_map_util

from ginjinn.core import Configuration

class AnnotationTypeError(Exception):
    ''' Error for wrong annotation type '''
    pass

class DatasetNotReadyError(Exception):
    pass

class TFDataset:
    ''' Tensorflow dataset object

        Object for the preparation of input data for the TF object detection API.
    '''
    def __init__(self, dataset_dir):
        dataset_path = Path(dataset_dir).resolve()

        self.config = Configuration({
            'dataset_dir': str(dataset_path),
            'dataset_json': str(dataset_path.joinpath('dataset.json').resolve()),
            'annotations_path': None,
            'annotation_type': None,
            'image_dir': None,
            'test_fraction': None,
            'csv_path': str(dataset_path.joinpath('annotations.csv').resolve()),
            'csv_train_path': str(dataset_path.joinpath('train.csv').resolve()),
            'csv_eval_path': str(dataset_path.joinpath('eval.csv').resolve()),
            'record_train_path': str(dataset_path.joinpath('train.record').resolve()),
            'record_eval_path': str(dataset_path.joinpath('eval.record').resolve()),
            'labelmap_path': str(dataset_path.joinpath('labelmap.pbtxt').resolve()),
            'ready': False,
        })
    
    def construct_dataset(self, annotations_path, image_dir, annotation_type, test_fraction):
        ''' 
            Build dataset from annotated images and split into test and evaluation datasets
        '''
        
        # update configuration
        annotation_type = annotation_type.lower()
        self.config.annotation_type = annotation_type
        self.config.annotations_path = str(Path(annotations_path).resolve())
        self.config.image_dir = str(Path(image_dir).resolve())
        self.config.test_fraction = test_fraction

        # create dataset directory if it does not exist
        Path(self.config.dataset_dir).mkdir(exist_ok=True)

        # build intermediary csv files
        if self.config.annotation_type == 'pascalvoc' or self.config.annotation_type == 'labelimg':
            xml_to_csv(self.config.annotations_path, self.config.csv_path)
        elif self.config.annotation_type == 'via':
            json_to_csv(self.config.annotations_path, self.config.csv_path)
        else:
            msg = f'Invalid annotation_type: {self.config.annotation_type}'
            raise AnnotationTypeError(msg)
        
        # build labelmap.pbtxt
        pbtxt_from_csv(self.config.csv_path, self.config.labelmap_path)

        # split csv in train and test
        generate_train_test(
            self.config.csv_path,
            self.config.csv_train_path,
            self.config.csv_eval_path,
            self.config.test_fraction
        )

        # build tensorflow record files
        generate_tfrecord(
            self.config.csv_train_path,
            self.config.labelmap_path,
            self.config.image_dir,
            self.config.record_train_path
        )
        generate_tfrecord(
            self.config.csv_eval_path,
            self.config.labelmap_path,
            self.config.image_dir,
            self.config.record_eval_path
        )

        # save dataset configuration
        self.config.ready = True
        self.to_json()

    def to_json(self, fpath=None):
        '''
            Write configuration json file
        '''
        fpath = fpath or self.config.dataset_json
        with open(fpath, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def load_json(self, fpath=None):
        '''
            Load configuration from json file
        '''
        fpath = fpath or self.config.dataset_json
        with open(fpath) as f:
            self.config = Configuration(json.load(f))

    def get_summary(self):
        '''
            Get dictionary with information about the dataset.
            This contains the directory, the readystate,
            number of images, number of objects, etc.
        '''
        return {
            'dataset_dir': self.config.dataset_dir,
            'ready': self.is_ready(),
            'n_images': self.n_images,
            'n_objects': self.n_objects,
            'classes': self.classes,
            'n_classes': self.n_classes,
            'n_samples_per_class': self.n_samples_per_class,
        }
    
    def is_ready(self):
        ''' returns whether the dataset is ready or not'''
        # if not self.config.ready:
        #     raise DatasetNotReadyError('Dataset is not ready. Run Dataset.construct_dataset first.')
        return self.config.ready

    @property
    def n_classes(self):
        if self.is_ready():
            return _get_n_classes_from_labelmap(self.config.labelmap_path)
        else:
            return None
    
    @property
    def classes(self):
        if self.is_ready():
            return list(_get_classdict_from_labelmap(self.config.labelmap_path).keys())
        else:
            return None
    
    @property
    def n_samples_per_class(self):
        if self.is_ready():
            return _get_samples_per_class_from_csv(self.config.csv_path)
        else:
            return None
    
    @property
    def n_images(self):
        if self.is_ready():
            return _get_n_images_from_csv(self.config.csv_path)
        else:
            return None

    @property
    def n_objects(self):
        if self.is_ready():
            return _get_n_samples_from_csv(self.config.csv_path)
        else:
            return None


def _get_classdict_from_labelmap(file_path):
    return label_map_util.get_label_map_dict(file_path, use_display_name=True)

def _get_n_classes_from_labelmap(file_path):
    return len(_get_classdict_from_labelmap(file_path))

def _get_samples_per_class_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df['class'].value_counts().to_dict()

def _get_n_samples_from_csv(file_path):
    df = pd.read_csv(file_path)
    return len(df.index)

def _get_n_images_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df['filename'].nunique()

'''
MIT License

Copyright (c) 2017 Dat Tran

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

'''
source: https://github.com/douglasrizzo/detection_util_scripts
'''

def __list_to_csv(annotations, output_file):
    column_name = [
        'filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'
    ]
    xml_df = pd.DataFrame(annotations, columns=column_name)
    xml_df.to_csv(output_file, index=None)


def xml_to_csv(xml_dir, output_file):
    """Reads all XML files, generated by labelImg, from a directory and generates a single CSV file"""
    annotations = []
    for xml_file in glob.glob(xml_dir + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text), member[0].text,
                     int(member[4][0].text), int(member[4][1].text),
                     int(member[4][2].text), int(member[4][3].text))
            annotations.append(value)

    __list_to_csv(annotations, output_file)


def json_to_csv(input_json, output_file):
    """Reads a JSON file, generated by the VGG Image Annotator, and generates a single CSV file"""
    with open(input_json) as f:
        images = json.load(f)

    annotations = []

    for entry in images:
        filename = images[entry]['filename']
        for region in images[entry]['regions']:
            c = region['region_attributes']['class']
            xmin = region['shape_attributes']['x']
            ymin = region['shape_attributes']['y']
            xmax = xmin + region['shape_attributes']['width']
            ymax = ymin + region['shape_attributes']['height']
            width = 0
            height = 0

            value = (filename, width, height, c, xmin, ymin, xmax, ymax)
            annotations.append(value)

    __list_to_csv(annotations, output_file)

def _pbtxt_from_classlist(l, pbtxt_path):
    pbtxt_text = ''

    for i, c in enumerate(l):
        pbtxt_text += 'item {\n    id: ' + str(
            i + 1) + '\n    display_name: "' + c + '"\n}\n\n'

    with open(pbtxt_path, "w+") as pbtxt_file:
        pbtxt_file.write(pbtxt_text)


def pbtxt_from_csv(csv_path, pbtxt_path):
    class_list = list(pd.read_csv(csv_path)['class'].unique())
    class_list.sort()

    _pbtxt_from_classlist(class_list, pbtxt_path)


def pbtxt_from_txt(txt_path, pbtxt_path):
    # read txt into a list, splitting by newlines
    data = [
        l.rstrip('\n').strip()
        for l in open(txt_path, 'r', encoding='utf-8-sig')
    ]

    data = [l for l in data if len(l) > 0]

    _pbtxt_from_classlist(data, pbtxt_path)

def generate_train_test(annotations_file, train_file, test_file, test_size=0.25):
    annotations = pd.read_csv(annotations_file)
    image_files = annotations['filename'].unique()

    train_image_files, test_image_files = train_test_split(image_files, test_size=test_size)

    annotations_train = annotations[annotations['filename'].isin(train_image_files)]
    annotations_test = annotations[annotations['filename'].isin(test_image_files)]

    annotations_train.to_csv(train_file, index=False)
    annotations_test.to_csv(test_file, index=False)

def __split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [
        data(filename, gb.get_group(x))
        for filename, x in zip(gb.groups.keys(), gb.groups)
    ]


def create_tf_example(group, path, class_dict):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)),
                        'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        if set(['xmin_rel', 'xmax_rel', 'ymin_rel', 'ymax_rel']).issubset(
                set(row.index)):
            xmin = row['xmin_rel']
            xmax = row['xmax_rel']
            ymin = row['ymin_rel']
            ymax = row['ymax_rel']

        elif set(['xmin', 'xmax', 'ymin', 'ymax']).issubset(set(row.index)):
            xmin = row['xmin'] / width
            xmax = row['xmax'] / width
            ymin = row['ymin'] / height
            ymax = row['ymax'] / height

        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_dict[row['class']])

    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/height':
                dataset_util.int64_feature(height),
                'image/width':
                dataset_util.int64_feature(width),
                'image/filename':
                dataset_util.bytes_feature(filename),
                'image/source_id':
                dataset_util.bytes_feature(filename),
                'image/encoded':
                dataset_util.bytes_feature(encoded_jpg),
                'image/format':
                dataset_util.bytes_feature(image_format),
                'image/object/bbox/xmin':
                dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax':
                dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin':
                dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax':
                dataset_util.float_list_feature(ymaxs),
                'image/object/class/text':
                dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label':
                dataset_util.int64_list_feature(classes),
            }))
    return tf_example


def class_dict_from_pbtxt(pbtxt_path):
    # open file, strip \n, trim lines and keep only
    # lines beginning with id or display_name
    data = [
        l.rstrip('\n').strip()
        for l in open(pbtxt_path, 'r', encoding='utf-8-sig')
        if 'id:' in l or 'display_name:'
    ]
    ids = [int(l.replace('id:', '')) for l in data if l.startswith('id')]
    names = [
        l.replace('display_name:', '').replace('"', '').strip() for l in data
        if l.startswith('display_name')
    ]

    print(data)

    # join ids and display_names into a single dictionary
    class_dict = {}
    for i in range(len(ids)):
        class_dict[names[i]] = ids[i]

    return class_dict

def generate_tfrecord(annotations_file, labelmap_file, image_dir, output_path):
    class_dict = class_dict_from_pbtxt(labelmap_file)

    writer = tf.io.TFRecordWriter(output_path)
    path = os.path.join(image_dir)
    examples = pd.read_csv(annotations_file)
    grouped = __split(examples, 'filename')

    for group in tqdm(grouped, desc='groups'):
        tf_example = create_tf_example(group, path, class_dict)
        writer.write(tf_example.SerializeToString())

    writer.close()
