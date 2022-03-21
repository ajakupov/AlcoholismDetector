import os

from beans.TestSetGenerator import TestSetGenerator
from beans.VideoInference import VideoInference

def generate_testset():
    root = "photos"
    for file in os.listdir(root):
        test_generator = TestSetGenerator(os.path.join(root, file))
        test_generator.capture_faces()

if __name__ == '__main__':
    video_inference = VideoInference()
    video_inference.run_stream()