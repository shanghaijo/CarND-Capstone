import numpy as np
import tensorflow as tf
from styx_msgs.msg import TrafficLight
import rospy
import cv2


class TLClassifier(object):
    PATH_TO_CKPT = 'light_classification/fine_tuned_model_real/frozen_inference_graph.pb'

    # Traffic Light Color mappings from REAL classifier labels
    # TL_COLOR = {
    #     1: TrafficLight.RED,
    #     2: TrafficLight.YELLOW,
    #     3: TrafficLight.GREEN,
    #     4: TrafficLight.UNKNOWN
    # }

    TL_COLOR = {
        2: TrafficLight.RED,
        3: TrafficLight.YELLOW,
        1: TrafficLight.GREEN,
        4: TrafficLight.UNKNOWN
    }

    TL_DEBUG = {
        TrafficLight.GREEN: "GREEN",
        TrafficLight.RED: "RED",
        TrafficLight.YELLOW: "YELLOW",
        TrafficLight.UNKNOWN: "UNKNOWN"
    }

    SCORE_THRESHOLD = 0.5

    def __init__(self, is_simulator):
        self.detection_graph = tf.Graph()

        rospy.loginfo("[TLClassifier] Loading: %s" % self.PATH_TO_CKPT)

        # Fix coutesy Anthony Sarkis (https://medium.com/@anthony_sarkis) taken from https://github.com/tensorflow/tensorflow/issues/6698
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                od_graph_def.ParseFromString(fid.read())
                tf.import_graph_def(od_graph_def, name='')
        self.sess = tf.Session(graph=self.detection_graph, config=config)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # print("get_classification")

        height, width, _ = image.shape

        boxes = scores = classes = None
        num_detections = 0

        img_expanded = np.expand_dims(image, axis=0)

        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})

        num_detections = int(num[0])

        detected_class = TrafficLight.UNKNOWN

        if num_detections > 0:
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)

            # print('boxes:\n%s\n' % boxes)
            # print('scores:\n%s\n' % scores)
            # print('classes:\n%s\n' % classes)

            # Find largest bounding box
            idx_max_square = -1
            max_square = 0

            for idx in range(scores.shape[0]):
                # Filter out low scores
                if scores[idx] >= self.SCORE_THRESHOLD:
                    box = boxes[idx]

                    box_height = abs((box[0] - box[2]) * height)
                    box_width = abs((box[1] - box[3]) * width)
                    square = box_height * box_width

                    if square > max_square:
                        max_square = square
                        idx_max_square = idx

            # print('largest box: %d (%d)\n' % (idx_max_square, max_square))

            if idx_max_square >= 0:
                # Retreive class and return
                detected_class = self.TL_COLOR[classes[idx_max_square]]

        rospy.loginfo("[TLClassifier] Detected traffic light: %s" % self.TL_DEBUG[detected_class])

        return detected_class
