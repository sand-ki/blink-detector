import cv2
import datetime
import dlib
import numpy as np
import pandas as pd
import time


def get_eye_aspect_ratio(eye_points):
    # Vertical eye landmarks
    left_vertical_distance = np.sqrt(np.sum(np.square(eye_points[1] - eye_points[5])))
    right_vertical_distance = np.sqrt(np.sum(np.square(eye_points[2] - eye_points[4])))

    # Horizontal eye landmarks
    horizontal_distance = np.sqrt(np.sum(np.square(eye_points[0] - eye_points[3])))

    # compute the EAR: (|(P2 - P6)| + |(P3 - P5)|) / 2 * |(P1 - P4)|
    ear = (left_vertical_distance + right_vertical_distance) / (2 * horizontal_distance)
    return ear


def main(ear_threshold: float = 0.3, min_frames_of_blinking: int = 3, frame_dimension: tuple = (640, 480),
         draw_contour: bool = True, video_writing: bool = False, fps: int = 60, output_folder: str = 'records/'):

    # setting up parameters
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('assets/shape_predictor_68_face_landmarks.dat')
    eye_points_dict = {"right_eye": list(range(36, 42)),
                       "left_eye": list(range(42, 48))}
    temp_blinking_counter = 0
    final_blinking_counter = 0
    previous_blinking_counter = 0

    vc = cv2.VideoCapture(1)
    if video_writing:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        file_path = f"{output_folder}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.avi"
        video_writer = cv2.VideoWriter(file_path, fourcc, fps, frame_dimension, True)

    start_time = time.time()
    output_list = list()

    while True:
        ret, img = vc.read()
        img = cv2.resize(img, frame_dimension)

        if ret:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray_img, 0)
            for face in faces:
                facial_landmarks = np.matrix([[p.x, p.y] for p in predictor(gray_img, face).parts()])
                left_eye_points = facial_landmarks[eye_points_dict.get("left_eye")]
                right_eye_points = facial_landmarks[eye_points_dict.get("right_eye")]

                if draw_contour:
                    left_eye_hull = cv2.convexHull(left_eye_points)
                    right_eye_hull = cv2.convexHull(right_eye_points)
                    cv2.drawContours(img, [left_eye_hull], -1, (36, 255, 12), 1)
                    cv2.drawContours(img, [right_eye_hull], -1, (36, 255, 12), 1)

                ear_left = get_eye_aspect_ratio(left_eye_points)
                ear_right = get_eye_aspect_ratio(right_eye_points)
                ear_mid = (ear_left + ear_right) / 2
                ear_mid_rounded = np.round(ear_mid * 100, 0)

                if ear_mid < ear_threshold:
                    temp_blinking_counter += 1
                else:
                    if temp_blinking_counter > min_frames_of_blinking:
                        final_blinking_counter += 1
                        print("Eye blinked")
                    temp_blinking_counter = 0

                cv2.putText(img, f'Blinks{final_blinking_counter}', (10, 30),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (36, 28, 237), 1)
                cv2.putText(img, f'EAR {ear_mid_rounded}', (10, 60),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (36, 28, 237), 1)

                if time.time() >= start_time + 60:
                    start_time = time.time()
                    blinking_in_time_range = final_blinking_counter - previous_blinking_counter
                    previous_blinking_counter = final_blinking_counter

                    minute = datetime.datetime.now().strftime('%Y%m%d-%H%M')
                    output_list.append({'time': minute,
                                        'blinking_rate': blinking_in_time_range})

            cv2.imshow("Winks Calculator", img)
            key = cv2.waitKey(1)
            if key is ord('q'):
                break
        if video_writing:
            video_writer.write(img)
    vc.release()
    cv2.destroyAllWindows()
    df_blinking = pd.DataFrame(output_list)
    df_blinking.to_csv(f"assets/data/blinking_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}.csv")


if __name__ == '__main__':
    output_folder = 'records/'
    fps = 30
    ear_threshold = 0.15
    min_frames_of_blinking = 1
    frame_dimension = (640, 480)
    draw_contour = True
    video_writing = False

    main(ear_threshold=ear_threshold, min_frames_of_blinking=min_frames_of_blinking, frame_dimension=frame_dimension,
         draw_contour=draw_contour, video_writing=video_writing, fps=fps, output_folder=output_folder)
