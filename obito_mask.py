import cv2
import mediapipe as mp
import numpy as np
from math import hypot
import matplotlib.pyplot as plt


mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


def get_face_points(face_landmarks, image_height, image_width):
    face_point_list = []
    for id, finger_list in enumerate(face_landmarks.landmark):
        fx = (finger_list.x) * image_width
        fy = (finger_list.y) * image_height
        fz = finger_list.z

        face_point_list.append([fx, fy, fz])

    return face_point_list


def get_noise_points(face_point_list):
    top_nose = (
        int(face_point_list[197][0]),
        int(face_point_list[197][1]),
    )
    center_nose = (
        int(face_point_list[195][0]),
        int(face_point_list[195][1]),
    )
    left_nose = (
        int(face_point_list[219][0]),
        int(face_point_list[219][1]),
    )
    right_nose = (
        int(face_point_list[439][0]),
        int(face_point_list[439][1]),
    )

    return top_nose, center_nose, left_nose, right_nose


def obito_filter():
    vidcap = cv2.VideoCapture(0)
    frame_width = int(vidcap.get(3))
    frame_height = int(vidcap.get(4))

    size = (frame_width * 2, frame_height * 2)
    # size = (1920, 1080)

    # Below VideoWriter object will create
    # a frame of above defined The output
    # is stored in 'filename.avi' file.
    result = cv2.VideoWriter(
        "filename.mp4", cv2.VideoWriter_fourcc(*"MP4V"), 20, size
    )
    with mp_face.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face:
        while True:
            drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
            ret, frame = vidcap.read()

            obito_path = "/home/iping-s/PP/face_filter/tobi.png"
            nose_image = cv2.imread(obito_path)
            # nose_image = cv2.cvtColor(obito, cv2.COLOR_BGR2RGB)

            image_height, image_width, _ = frame.shape

            nose_mask = np.ones((image_height, image_width), np.uint8)

            # image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            image = cv2.flip(frame, 1)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_height, image_width, _ = image.shape
            results = face.process(image)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    face_point_list = get_face_points(
                        face_landmarks, image_height, image_width
                    )

                    (
                        top_nose,
                        center_nose,
                        left_nose,
                        right_nose,
                    ) = get_noise_points(face_point_list)

                    nose_width = int(
                        hypot(
                            left_nose[0] - right_nose[0],
                            left_nose[1] - right_nose[1],
                        )
                        * 4.7
                    )

                    nose_height = int(nose_width * 1.5)

                    top_left = (
                        int(center_nose[0] - nose_width / 2),
                        int(center_nose[1] - nose_height / 2),
                    )
                    bottom_right = (
                        int(center_nose[0] + nose_width / 2),
                        int(center_nose[1] + nose_height / 2),
                    )

                    try:

                        nose_pig = cv2.resize(
                            nose_image, (nose_width, nose_height)
                        )
                        nose_pig_gray = cv2.cvtColor(
                            nose_pig, cv2.COLOR_BGR2GRAY
                        )
                        _, nose_mask = cv2.threshold(
                            nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV
                        )

                        nose_area = image[
                            top_left[1] : top_left[1] + nose_height,
                            top_left[0] : top_left[0] + nose_width,
                        ]

                        nose_area_no_nose = cv2.bitwise_and(
                            nose_area, nose_area, mask=nose_mask
                        )
                        final_nose = cv2.add(nose_area_no_nose, nose_pig)

                        image[
                            top_left[1] : top_left[1] + nose_height,
                            top_left[0] : top_left[0] + nose_width,
                        ] = final_nose
                    except Exception as e:
                        image = image

                    """
                    mp_drawing.draw_landmarks(
                        image,
                        face_landmarks,
                        mp_face.FACE_CONNECTIONS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec,
                    )
                    """
            cv2.imshow("frame", image)
            image = cv2.resize(
                image, (frame_width * 2, frame_height * 2), cv2.INTER_CUBIC
            )
            result.write(image)

            # cv2.convexHull()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    vidcap.release()
    result.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    obito_filter()
