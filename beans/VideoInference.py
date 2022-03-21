from math import ceil, floor
import cv2
import tensorflow as tf
import numpy as np
from xgboost import cv
from helpers.face_helper import get_faces
from helpers.image_helpers import image_resize

class VideoInference:
    # BGR color constants
    WHITE = (255, 255, 255)
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLACK = (0, 0, 0)

    WINDOW_NAME = "Stay Sober"

    output_layer = 'loss:0'
    input_node = 'Placeholder:0'
    predicted_tag = 'Predicted Tag'

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 1


    def __init__(self) -> None:
        prototxt = "./ml_artifacts/deploy.prototxt.txt"
        face_model = "./ml_artifacts/res10_300x300_ssd_iter_140000.caffemodel"
        
        self.graph_def = tf.compat.v1.GraphDef()
        # list of classes
        self.labels = ['alcoholic', 'sober']

        with tf.io.gfile.GFile(name='./ml_artifacts/model.pb', mode='rb') as f:
            self.graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def=self.graph_def, name='')

        self.net = cv2.dnn.readNetFromCaffe(prototxt, face_model)

    def run_stream(self):
        video_capture = cv2.VideoCapture(0)
        with tf.compat.v1.Session() as sess:
            prob_tensor = sess.graph.get_tensor_by_name(self.output_layer)
            while(video_capture.isOpened()):
                # read video frame by frame
                _, frame = video_capture.read()
                frame = cv2.flip(frame, 1)

                faces = get_faces(frame, self.net)

                for (startX, startY, endX, endY) in faces:
                    face_frame = frame[startY:endY, startX:endX]

                    input_tensor_shape = sess.graph.get_tensor_by_name(
                        self.input_node).shape.as_list()
                    network_input_size = input_tensor_shape[1]

                    face_frame = cv2.resize(face_frame, (network_input_size, network_input_size), interpolation=cv2.INTER_LINEAR)

                    predictions = sess.run(
                    prob_tensor, {self.input_node: [face_frame]})

                    # get the highest probability label
                    highest_probability_index = np.argmax(predictions)
                    predicted_tag = self.labels[highest_probability_index]
                    highest_probability = max(max(predictions)) * 100
                    highest_probability = ceil(highest_probability)
                    output_text = "{} {}%".format(predicted_tag, highest_probability)

                    if predicted_tag == 'sober':
                        frame_color = self.GREEN
                    elif predicted_tag == 'alcoholic':
                        frame_color = self.RED
                    else:
                        frame_color = self.BLUE

                    cv2.rectangle(frame, (startX, startY), (endX, endY), frame_color, 2)
                    cv2.putText(
                        frame, 
                        output_text, 
                        (startX, endY), 
                        self.font, 
                        self.fontScale, 
                        frame_color, 
                        self.thickness)

                cv2.imshow(self.WINDOW_NAME, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        video_capture.release()
        cv2.destroyAllWindows()