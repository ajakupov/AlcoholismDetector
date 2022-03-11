import os
from beans.TestSetGenerator import TestSetGenerator

if __name__ == '__main__':
    root = "photos"
    for file in os.listdir(root):
        test_generator = TestSetGenerator(os.path.join(root, file))
        test_generator.capture_faces()