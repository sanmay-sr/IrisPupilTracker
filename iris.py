import cv2 as cv
import numpy as np
import mediapipe as mp
import time

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Define eye and iris landmarks
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Open webcam
cap = cv.VideoCapture(0)
time.sleep(2)  # Allow camera stabilization

# Capture a single frame and release the camera
ret, frame = cap.read()
cap.release()
cv.destroyAllWindows()

if ret:
    # Flip the image and process it with Mediapipe
    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        img_h, img_w = frame.shape[:2]
        mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

        def process_iris(iris_landmarks, eye_side):
            # Extract iris center and radius
            (cx, cy), iris_radius = cv.minEnclosingCircle(mesh_points[iris_landmarks])
            iris_center = np.array([cx, cy], dtype=np.int32)
            iris_radius = int(iris_radius)

            # Define ROI around the iris and crop the iris image
            iris_roi = frame[iris_center[1] - iris_radius:iris_center[1] + iris_radius,
                             iris_center[0] - iris_radius:iris_center[0] + iris_radius]

            if iris_roi.size != 0:
                # Convert iris region to grayscale
                iris_gray = cv.cvtColor(iris_roi, cv.COLOR_BGR2GRAY)

                # Apply a mask to focus on the center of the iris
                mask = np.zeros_like(iris_gray)
                cv.circle(mask, (iris_radius, iris_radius), iris_radius, 255, -1)  # Iris mask
                iris_gray_masked = cv.bitwise_and(iris_gray, iris_gray, mask=mask)

                # Adaptive thresholding to isolate the pupil (darkest region in the center)
                pupil_thresh = cv.adaptiveThreshold(iris_gray_masked, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv.THRESH_BINARY_INV, 11, 2)
                contours, _ = cv.findContours(pupil_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                # Find the best candidate for the pupil
                valid_pupil_detected = False
                pupil_radius = 0
                pupil_center = (iris_radius, iris_radius)  # Start with iris center as default

                if contours:
                    for contour in contours:
                        (pupil_x, pupil_y), radius = cv.minEnclosingCircle(contour)
                        radius = int(radius)

                        # Ensure the pupil is within the center and is reasonably small
                        distance_to_center = np.linalg.norm([pupil_x - iris_radius, pupil_y - iris_radius])
                        if radius <= iris_radius // 2 and distance_to_center <= iris_radius // 4:
                            valid_pupil_detected = True
                            pupil_center = (int(pupil_x), int(pupil_y))
                            pupil_radius = radius
                            break

                if valid_pupil_detected:
                    # Calculate the iris-to-pupil ratio
                    iris_area = np.pi * (iris_radius ** 2)
                    pupil_area = np.pi * (pupil_radius ** 2)
                    iris_pupil_ratio = iris_area / pupil_area if pupil_area > 0 else 0

                    # Output results
                    print(f"{eye_side} Iris Radius: {iris_radius}")
                    print(f"{eye_side} Pupil Radius: {pupil_radius}")
                    print(f"{eye_side} Iris-to-Pupil Ratio: {iris_pupil_ratio:.2f}")

                    # Annotate the iris with a red circle and the pupil with a green circle
                    cv.circle(iris_roi, (iris_radius, iris_radius), iris_radius, (0, 0, 255), 1)  # Red for iris
                    cv.circle(iris_roi, pupil_center, pupil_radius, (0, 255, 0), 1)  # Green for pupil

                    # Save the annotated iris image
                    cv.imwrite(f'{eye_side.lower()}_annotated_iris.png', iris_roi)
                else:
                    print(f"Failed to detect a valid pupil for {eye_side}.")
            else:
                print(f"Failed to extract {eye_side} iris region.")

        # Process both eyes
        process_iris(LEFT_IRIS, "Left Eye")
        process_iris(RIGHT_IRIS, "Right Eye")
    else:
        print("No face landmarks detected.")
else:
    print("Failed to capture image from camera.")