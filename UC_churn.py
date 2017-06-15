# pip install keras

import keras
from keras.models import Sequential
from keras.layers import Dense

from keras import backend as K
import tensorflow as tf


# prepare input data
X = []
y = []

with open('./UC_churn_input.csv', 'r') as input_file:
    for line in input_file.readlines():
        id,feature_vector,target = line.strip().split('","')
        id = id[1:]

        vector = []
        for dimension in feature_vector[1:-1].split(','):
            vector.append(float(dimension))
        # vector = [float(dimension) for dimension in feature_vector[1:-1].split(',')]

        X.append(vector)
        y.append(int(target[:-1]))

# build dense network
model = Sequential()
model.add(Dense(22, input_dim = 19, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'binary_crossentropy'])

model.fit(X, y, epochs=300, batch_size=10)

K.set_learning_phase(0)
config = model.get_config()
weights = model.get_weights()

export_model = Sequential.from_config(config)
export_model.set_weights(weights)

export_dir = './model_demo/'

with K.get_session() as session:
    input_tensor = tf.get_default_graph().get_tensor_by_name("dense_1_input:0")
    output_tensor = tf.get_default_graph().get_tensor_by_name("dense_4_target:0")

    input_mapping = {'input':export_model.input}
    output_mapping = {'output':export_model.output}

    signature_definition = tf.saved_model.signature_def_utils.predict_signature_def(inputs=input_mapping, outputs=output_mapping)
    signature_definition_map = {tf.saved_model.signature_constants.PREDICT_METHOD_NAME:signature_definition}

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(session, [ tf.saved_model.tag_constants.SERVING], signature_def_map= signature_definition_map)
    builder.save(True)