from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf
from model import select_model, get_checkpoint
from utils import *
import os
import re
from tensorflow.contrib.layers import *
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base
import urllib
import boto3
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

RESIZE_FINAL = 227
GENDER_LIST =['M','F']
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
MAX_BATCH_SZ = 128

#MODEL_GRAPH_DEF_PATH = os.path.join(os.sep, 'tmp', 'agemodel.pb')
#MODEL_GRAPH_DEF_PATH = './age_graph.pb'

MODEL_GRAPH_DEF_PATH = os.path.join(os.sep, 'tmp', 'model.pb')

# Feel free to load these as environment variables through Lambda.
LABEL_STRINGS_FILENAME = os.path.join(
    'imagenet', 'imagenet_2012_challenge_label_map_proto.pbtxt')
LABEL_IDS_FILENAME = os.path.join(
    'imagenet', 'imagenet_synset_to_human_label_map.txt')

MODEL_FILENAME = 'age_graph.pb'

SERVICE = os.environ['SERVICE']

rekognition = boto3.client('rekognition')

class ReturnResult:
  IsExplicit = None
  IsChild = None
  AgeRange = None
  AgeRangeConfidence = None

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(MODEL_GRAPH_DEF_PATH, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ =tf.import_graph_def(graph_def,name='')

def one_of(fname, types):
    return any([fname.endswith('.' + ty) for ty in types])

def resolve_file(fname):
    if os.path.exists(fname):
      return fname
    for suffix in ('.jpg', '.png', '.JPG', '.PNG', '.jpeg'):
        cand = fname + suffix
        if os.path.exists(cand):
            return cand
    return None

# Creates node ID --> English string lookup.
print('Loading ID-to-string dict from file...')
NODE_LOOKUP = NodeLookup()

# This must be called before create_graph().
print('Downloading Model from S3...')
s3.Bucket(os.environ['model_bucket_name']).download_file(
                                                      MODEL_FILENAME,
                                                      MODEL_GRAPH_DEF_PATH)

# Creates graph from saved GraphDef.
print('Creating TF computation graph...')
create_graph()

def classify_many_single_crop(sess, label_list, softmax_output, coder, images, image_data):
    result = -1
    is_child =False
    percentage = 0
    try:
        image_batch = make_single_multi_crop_batch(image_data, coder)
        batch_results = sess.run(softmax_output, feed_dict={images:image_batch.eval()})
        batch_sz = batch_results.shape[0]
        for i in range(batch_sz):
          output_i = batch_results[i]
          best_i = np.argmax(output_i)
          best_choice = (label_list[best_i], output_i[best_i])
                
          if float(output_i[best_i])>percentage:
           percentage = float(output_i[best_i])
           result = int(best_i)
    except Exception as e:
        print(e)
        print('Failed to run all images')
    is_child = True if result>-1 and result < 3 else False
    label =  label_list[result] if result>-1 and result < 3 else ''
    return is_child,percentage,label


def run_inference_on_image(image_data):
    config = tf.ConfigProto(allow_soft_placement=True)
    tf.logging.set_verbosity(tf.logging.WARN)
    with tf.Session(config=config) as sess:
        
        softmax_output =sess.graph.get_tensor_by_name('labels_softmax:0')
        images = sess.graph.get_tensor_by_name('Placeholder:0')
        label_list = AGE_LIST
        nlabels = len(label_list)
        model_fn = select_model('inception_v3')

        #print('Executing on %s' % FLAGS.device_id)
        image_bytes = download_image(url)
        coder = ImageCoder()
        return classify_many_single_crop(sess, label_list, softmax_output, coder, images, image_bytes)


def lambda_handler(event, context):
   
    record = event['Record']
    bucket = record['s3']['bucket']['name']
    key = record['s3']['object']['key']
    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, 'wb') as f:
      s3.Bucket(bucket).download_file(key, tmp.name)
      tmp.flush()

    
    image_bytes = fetchImgData(tmp.name)
    print('Checking image for explicit content...')
    
    is_explicit = detect_explicit_content(image_bytes)
    is_child,percentage,label = run_inference_on_image(image_bytes)
    mReturnResult = ReturnResult()
    mReturnResult.IsExplicit = str(is_explicit)
    mReturnResult.IsChild = str(is_child)
    mReturnResult.AgeRange = str(label)
    mReturnResult.AgeRangeConfidence = str(percentage)
    return json.dumps(mReturnResult.__dict__)

def fetchImgData(image):
  image_data = tf.gfile.FastGFile(image, 'rb').read()
  return image_data
def download_image(url):
    """ Download image from private Slack URL using bearer token authorization.

    Args:
        url (string): Private Slack URL for uploaded image.

    Returns:
        (bytes)
        Blob of bytes for downloaded image.


    """
    request = urllib.request.Request(url, headers={'Authorization': 'Bearer %s' % ACCESS_TOKEN})
    return urllib.request.urlopen(request).read()


def detect_explicit_content(image_bytes):
    """ Checks image for explicit or suggestive content using Amazon Rekognition Image Moderation.

    Args:
        image_bytes (bytes): Blob of image bytes.

    Returns:
        (boolean)
        True if Image Moderation detects explicit or suggestive content in blob of image bytes.
        False otherwise.

    """
    try:
        response = rekognition.detect_moderation_labels(
            Image={
                'Bytes': image_bytes,
            },
            MinConfidence=50
        )
    except Exception as e:
        print(e)
        print('Unable to detect labels for image.')
        raise(e)
    labels = response['ModerationLabels']
    is_explicit = False
    for modLabel in response['ModerationLabels']:
      if((modLabel['ParentName'] == 'Explicit Nudity' or modLabel['Name'] == 'Explicit Nudity') and Decimal(str(modLabel['Confidence'])) >= 70):
        is_explicit = True
    return is_explicit


def delete_file(file_id):
    """ Deletes file from Slack team via Slack API.

    Args:
        file_id (string): ID of file to delete.

    Returns:
        (None)
    """
    url = 'https://slack.com/api/files.delete'
    data = urllib.parse.urlencode(
        (
            ("token", ACCESS_TOKEN),
            ("file", file_id)
        )
    )
    data = data.encode("ascii")
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    request = urllib.request.Request(url, data, headers)
    urllib.request.urlopen(request)


#if __name__ == '__main__':
#    tf.app.run()
