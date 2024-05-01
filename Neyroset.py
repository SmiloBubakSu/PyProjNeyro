import os

from imageai.Detection import VideoObjectDetection

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
detector.loadModel(detection_speed="fast")

######## Video Analitic ########
def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    print("Секунди : ", second_number)
    print("Масив для вихідних даних кожного кадру", output_arrays)
    print("Масив для кількості вихідних даних для унікальних об'єктів у кожному кадрі : ", count_arrays)
    print("Вивести середню кількість унікальних об’єктів за останню секунду: ", average_output_count)
    print("------------ Кінець Секунди --------------")
###################################
#Запис результата в текстовий формат
    for second in str(average_output_count):
        second = str(second_number)
        if second_number != 0:
            a = "\n" + second + ": " + "\t" + str(average_output_count)
            break
        else:
            print("Error")
    with open("result.txt", "a") as file:
            file.write(a)
###################################
video_path = detector.detectObjectsFromVideo(
    input_file_path=os.path.join(execution_path, "Car1.mp4"), #Вхідним є наше відео з інтернета в форматі мр4
    output_file_path=os.path.join(execution_path, "traffic_detected"), #Вихідним результат після обробки нейронної мережі
    frames_per_second=1,
    per_second_function=forSeconds,
    minimum_percentage_probability=30,
)
####################################
