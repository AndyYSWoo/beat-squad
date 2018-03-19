# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains the entrypoint to the rest of the code"""

from __future__ import absolute_import
from __future__ import division

import os
import io
import json
import sys
import logging
import tensorflow as tf
from collections import Counter
import json

# Extra options for ensemble
tf.app.flags.DEFINE_string("json_in_path", "", "For official_eval mode, path to JSON input file. You need to specify this for official_eval_mode.")
tf.app.flags.DEFINE_string("ensemble_dir", "", "Path to directory with models. Without slash. Stronger model comes first. ")
tf.app.flags.DEFINE_string("ensemble_folder_names", "", "Folder_name for each model. Split by comma.")
tf.app.flags.DEFINE_string("ensemble_class_names", "", "Class name for each model. Split by comma.")
tf.app.flags.DEFINE_string("hidden_size_list", "", "Size of the hidden states")
tf.app.flags.DEFINE_string("context_len_list", "", "The maximum context length of your model")
tf.app.flags.DEFINE_string("embedding_size_list", "", "Size of the pretrained word vectors. This needs to be one of the available GloVe dimensions: 50/100/200/300")
tf.app.flags.DEFINE_string("glove_path", "", "Path to glove .txt file. Defaults to data/glove.6B.{embedding_size}d.txt")

tf.app.flags.DEFINE_string("ensemble_algorithm", "voting", "")
tf.app.flags.DEFINE_string("json_out_path", "predictions.json", "Output path for official_eval mode. Defaults to predictions.json")

FLAGS = tf.app.flags.FLAGS

# Example usage:
#
# Local:
# python code/main_ensemble.py \
# --json_in_path=data/tiny-dev.json \
# --ensemble_dir=experiments \
# --ensemble_folder_names=fulldp,deepencoder \
# --ensemble_class_names=RNetPtrModel,RNetPtrDeepModel \
# --hidden_size_list=200,200 \
# --context_len_list=450,450 \
# --embedding_size_list=300,300
#
# Codalab:
# cl run --name gen-answers --request-docker-image abisee/cs224n-dfp:v4 :code :experiments glove.txt:0x97c870/glove.6B.300d.txt data.json:0x4870af 'python main_ensemble.py --mode=official_eval --glove_path=glove.txt --json_in_path=data.json --ensemble_dir=experiments --ensemble_folder_names=fulldp,deepencoder --ensemble_class_names=RNetPtrModel,RNetPtrDeepModel --hidden_size=200,200 --context_len=450,450 --embedding_size=300,300 ' --request-memory 6g

def main(unused_argv):
    ensemble_folder_names = FLAGS.ensemble_folder_names.split(',')
    ensemble_class_names = FLAGS.ensemble_class_names.split(',')
    hidden_size_list = FLAGS.hidden_size_list.split(',')
    context_len_list = FLAGS.context_len_list.split(',')
    embedding_size_list = FLAGS.embedding_size_list.split(',')
    num_model = len(ensemble_folder_names)
    ensemble_dir = FLAGS.ensemble_dir
    if FLAGS.ensemble_dir != "" and FLAGS.ensemble_dir[-1] == '/':
        ensemble_dir = FLAGS.ensemble_dir[:-1]

    for i in range(num_model):
        command = 'python code/main.py'
        command += ' --hidden_size=' + hidden_size_list[i]
        command += ' --context_len=' + context_len_list[i]
        command += ' --embedding_size=' + embedding_size_list[i]
        command += ' --mode=official_eval'
        command += ' --json_in_path=' + FLAGS.json_in_path
        command += ' --ckpt_load_dir=' + ensemble_dir + '/' + ensemble_folder_names[i] + '/best_checkpoint'
        command += ' --class_name=' + ensemble_class_names[i]
        if FLAGS.glove_path != '':
            command += ' --glove_path=' + FLAGS.glove_path
        print '[LOG]', command

        os.system(command)
        os.rename('predictions.json', 'predictions_' + ensemble_folder_names[i] + '.json')

    prediction = {}
    for i in range(num_model):
        prediction_file = 'predictions_' + ensemble_folder_names[i] + '.json'
        f = open(prediction_file)
        current = json.load(f)
        for ID in current:
            if ID not in prediction:
                prediction[ID] = []
            prediction[ID].append(current[ID])

    if FLAGS.ensemble_algorithm == 'voting':
        for ID in prediction:
            prediction[ID] = Counter(prediction[ID]).most_common(1)[0][0]

    # Write the uuid->answer mapping a to json file in root dir
    print "Writing predictions to %s..." % FLAGS.json_out_path
    with io.open(FLAGS.json_out_path, 'w', encoding='utf-8') as f:
        f.write(unicode(json.dumps(prediction, ensure_ascii=False)))
        print "Wrote predictions to %s" % FLAGS.json_out_path


if __name__ == "__main__":
    tf.app.run()
