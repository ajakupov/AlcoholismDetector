from beans.TrainSetGenerator import TrainSetGenerator

if __name__ == '__main__':
    generator = TrainSetGenerator("https://www.youtube.com/watch?v=cFiJOgB56vk")
    generator.capture_faces()
