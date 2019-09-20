import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw
from os.path import isdir, abspath, join, isfile, exists
import os
from os.path import join, abspath, isdir, exists

from object_detection.utils import label_map_util

# TODO: rework this to match new version's style, e.g. pathlib,...
class TFDetector(object):
    ''' ginjinn TFDetector object

        Class for detection on objects in images based on a
        pretrained tensorflow model checkpoint and the corresponding
        labelmap.pbtxt
    '''
    def __init__(self, frozen_model_path, labelmap_path):
        self.frozen_model_path = frozen_model_path
        self.labelmap_path = labelmap_path
        self.detection_graph = self._build_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True # pylint: disable=no-member
        self.sess = tf.Session(graph=self.detection_graph, config=config)

        label_map = label_map_util.load_labelmap(self.labelmap_path)
        categories = label_map_util.convert_label_map_to_categories(
                label_map, max_num_classes=90, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def _build_graph(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default(): # pylint: disable=not-context-manager
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.frozen_model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        return detection_graph

    def _load_image_into_numpy_array(self, image):
        return np.array(image).astype(np.uint8)

    def detect(self, image):
        '''
            Detect objects on a single image
        '''
        image_np = self._load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)

        graph = self.detection_graph
        image_tensor = graph.get_tensor_by_name('image_tensor:0')
        boxes = graph.get_tensor_by_name('detection_boxes:0')
        scores = graph.get_tensor_by_name('detection_scores:0')
        classes = graph.get_tensor_by_name('detection_classes:0')
        num_detections = graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = self.sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

        boxes, scores, classes, num_detections = map(
                np.squeeze, [boxes, scores, classes, num_detections])

        return boxes, scores, classes.astype(int), num_detections

    def run_detection(self, out_dir, image_path, output_types, padding=0, th=0.5):
        '''
            Run detection and save outputs to files

            Parameters
            ----------
            out_dir : string
                path to output directory
            image_path: string
                path to single image or directory containing images
            output_type: list
                list of output types ['ibb', 'ebb', 'csv']
            padding: int
                padding to apply to bounding boxes in pixel
            th: float
                score threshold to still consider a box. Boxes with scores
                below th will not be present in the output
        '''

        # get image files
        out_dir = abspath(out_dir)
        image_path = abspath(image_path)
        if not isdir(image_path):
            img_files = [image_path]
        else:
            extensions = ['.jpg', '.jpeg', '.png']
            img_files = [
                join(image_path, f) for f in os.listdir(image_path) if (any(f.endswith(ext) for ext in extensions))
            ]

        if not exists(out_dir):
            os.mkdir(out_dir)

        # detect
        for f in img_files:
            bname, ext = os.path.splitext(os.path.basename(f))

            image = Image.open(f).convert('RGB')
            boxes, scores, classes, num_detections = self.detect(image)

            cls_map = self.get_classmap(classes)
            
            idcs = [i for i, score in enumerate(scores) if score >= th]

            cur_dir = join(out_dir, bname)
            if not exists(cur_dir):
                os.mkdir(cur_dir)

            if 'ibb' in output_types:
                img_dir = join(cur_dir, 'ibb')
                if not exists(img_dir):
                    os.mkdir(img_dir)
                for c, c_n in cls_map.items():
                    new_img = image.copy()
                    _boxes = [box for i, box in enumerate(boxes) if (i in idcs) and (classes[i] == c)]
                    for box in _boxes:
                        draw_bounding_box_on_image(new_img, box, color='red', thickness=5)

                    new_img.save(
                        join(img_dir, '{}_{}{}'.format(bname, c_n, ext))
                    )
            
            if 'ebb' in output_types:
                bb_dir = join(cur_dir, 'ebb')
                if not exists(bb_dir):
                    os.mkdir(bb_dir)
                for i, box in enumerate(boxes):
                    if not (i in idcs):
                        continue
                    bb_image = extract_bounding_box_from_image(image, box, padding=padding)
                    class_name = cls_map[classes[i]]
                    bb_image.save(
                        join(bb_dir, '{}_{}_{}{}'.format(bname, i, class_name, ext))
                    )
            
            if 'csv' in output_types:
                filtered_boxes = [box for i, box in enumerate(boxes) if i in idcs]
                filtered_scores = [score for i, score in enumerate(scores) if i in idcs]
                boxes_to_csv(filtered_boxes, image, join(cur_dir, 'boxes.csv'), scores=filtered_scores)

    def get_classmap(self, classes):
        return {
            c: self.category_index[c]['name'] for c in set(classes)
        }


def boxes_to_csv(boxes, image, out_path, scores=None):
    boxes = [(*_rel_box_to_img_coords(box, image), *box) for box in boxes]
    df = pd.DataFrame(
        boxes,
        columns=[
            'ymin', 'xmin', 'ymax', 'xmax', 'ymin_rel', 'xmin_rel', 'ymax_rel', 'xmax_rel',
        ]
    )
    if scores:
        df['score'] = scores
    df.to_csv(out_path, index=False)

def _rel_box_to_img_coords(box, image):
    im_width, im_height = image.size
    ymin, xmin, ymax, xmax = box
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                                                ymin * im_height, ymax * im_height)
    return (left, right, top, bottom)

def draw_bounding_box_on_image(image, box, color='red', thickness=4):
    (left, right, top, bottom) = _rel_box_to_img_coords(box, image)
    
    draw = ImageDraw.Draw(image)
    draw.line([(left, top), (left, bottom), (right, bottom),
                         (right, top), (left, top)], width=thickness, fill=color)

def extract_bounding_box_from_image(image, box, padding=0):
        im_width, im_height = image.size
        (left, right, top, bottom) = _rel_box_to_img_coords(box, image)

        left, right = np.clip((left - padding, right + padding), 0, im_width - 1)
        top, bottom = np.clip((top - padding, bottom + padding), 0, im_height - 1)

        return image.crop((left, top, right, bottom))
