from keras.models import Sequential
from keras.layers import Dense
from keras import backend as k
import tensorflow as tf
import numpy as np


_input_file = "./inputData/UC_churn_input.csv"
_output_model_directory = "./models/model_UC_churn/"
_print_graph_operations = False


def read_csv(file):
    x_input = []
    y_input = []
    with open(file, 'r') as input_file:
        for line in input_file.readlines()[1:]:
            feature_vector, target = line.strip().split('","')
            feature_vector = feature_vector[1:]
            vector = []
            for dimension in feature_vector[1:-1].split(','):
                vector.append(float(dimension))
            x_input.append(vector)
            y_input.append(int(target[:-1]))
    return np.array(x_input), np.array(y_input)


def save_model(export_dir, print_graph_operations=False):
    k.set_learning_phase(0)
    config = model.get_config()
    weights = model.get_weights()
    export_model = Sequential.from_config(config)
    export_model.set_weights(weights)
    with k.get_session() as session:
        input_mapping = {'input': export_model.input}
        output_mapping = {'output': export_model.output}
        signature_definition = tf.saved_model.signature_def_utils.predict_signature_def(inputs=input_mapping,
                                                                                        outputs=output_mapping)
        signature_definition_map = {tf.saved_model.signature_constants.PREDICT_METHOD_NAME: signature_definition}
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map=signature_definition_map)
        builder.save(True)
        if print_graph_operations:
            print("Graph operations: >>", session.graph.get_operations(), "<<")


# read data
X, Y = read_csv(_input_file)

# create, compile, fit and evaluate model
model = Sequential()
model.add(Dense(30, input_dim=19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=150, batch_size=10)
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# save model
save_model(_output_model_directory, _print_graph_operations)
