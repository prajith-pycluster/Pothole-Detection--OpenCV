import cv2 as cv
import os
import math
import csv

# ================= LOAD YOLO MODEL =================

net = cv.dnn.readNet(
    'utils/yolov4_tiny.weights',
    'utils/yolov4_tiny.cfg'
)

net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

model = cv.dnn_DetectionModel(net)

model.setInputParams(
    size=(416, 416),
    scale=1/255,
    swapRB=True
)

# ================= VIDEO LIST =================

videos = [

    {
        "name": "test.mp4",
        "actual": 85
    },

    {
        "name": "test1.mp4",
        "actual": 20
    },

    {
        "name": "test2.mp4",
        "actual": 26
    },

    {
        "name": "test3.mp4",
        "actual": 13
    }
]

# ================= PARAMETERS =================

CONFIDENCE = 0.5
NMS = 0.4

# ================= RESULT STORAGE =================

all_results = []

accuracy_list = []

# ================= PROCESS EACH VIDEO =================

for video in videos:

    print("\n====================================")
    print(f"PROCESSING : {video['name']}")
    print("====================================")

    cap = cv.VideoCapture(video["name"])

    width = cap.get(3)
    height = cap.get(4)

    tracked_potholes = []

    pothole_id = 0
    total_potholes = 0

    frame_count = 0

    # ================= VIDEO LOOP =================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # ================= DETECTION =================

        classes, scores, boxes = model.detect(
            frame,
            CONFIDENCE,
            NMS
        )

        current_frame_potholes = 0

        for (classid, score, box) in zip(classes, scores, boxes):

            if score < 0.75:
                continue

            x, y, w, h = box

            center_x = x + w // 2
            center_y = y + h // 2

            rec_area = w * h
            frame_area = width * height

            # ================= SEVERITY =================

            severity = "Low"

            if rec_area / frame_area > 0.03:
                severity = "High"

            elif rec_area / frame_area > 0.01:
                severity = "Medium"

            # ================= COLORS =================

            if severity == "High":
                color = (0, 0, 255)

            elif severity == "Medium":
                color = (0, 255, 255)

            else:
                color = (0, 255, 0)

            # ================= TRACKING =================

            matched = False

            for pothole in tracked_potholes:

                px, py, pw, ph = pothole["box"]

                prev_center_x = px + pw // 2
                prev_center_y = py + ph // 2

                center_distance = math.sqrt(
                    (center_x - prev_center_x) ** 2 +
                    (center_y - prev_center_y) ** 2
                )

                size_difference = abs(
                    (w * h) - (pw * ph)
                )

                # SAME POTHOLE
                if center_distance < 70 and size_difference < 15000:

                    pothole["box"] = (x, y, w, h)

                    pothole["severity"] = severity

                    pothole["last_seen"] = frame_count

                    pothole_id_display = pothole["id"]

                    matched = True

                    break

            # ================= NEW POTHOLE =================

            if not matched:

                pothole_id += 1

                pothole_id_display = pothole_id

                tracked_potholes.append({

                    "id": pothole_id,

                    "box": (x, y, w, h),

                    "severity": severity,

                    "last_seen": frame_count
                })

            # ================= DRAW BOX =================

            cv.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            cv.putText(
                frame,
                f"ID: {pothole_id_display}",
                (x, y - 35),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            cv.putText(
                frame,
                f"{severity}",
                (x, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            current_frame_potholes += 1

        # ================= REMOVE OLD TRACKS =================

        active_tracks = []

        for pothole in tracked_potholes:

            if frame_count - pothole["last_seen"] > 30:

                total_potholes += 1

                pid = pothole["id"]

                severity = pothole["severity"]

                x, y, w, h = pothole["box"]

                print(
                    f"Pothole #{pid} | "
                    f"Severity: {severity} | "
                    f"Position: ({x}, {y})"
                )

            else:

                active_tracks.append(pothole)

        tracked_potholes = active_tracks

        # ================= ROAD SCORE =================

        road_score = 100 - (current_frame_potholes * 5)

        road_score = max(0, min(100, road_score))

        # ================= DISPLAY =================

        cv.putText(
            frame,
            f"Total Potholes: {total_potholes}",
            (20, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv.putText(
            frame,
            f"Road Health Score: {road_score}%",
            (20, 90),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv.imshow(video["name"], frame)

        key = cv.waitKey(30)

        if key == ord('q'):
            break

    # ================= FINAL VIDEO CALCULATION =================

    actual = video["actual"]

    detected = total_potholes

    # correct detections
    correct = min(actual, detected)

    # false detections
    false_detections = max(0, detected - actual)

    # balanced accuracy
    accuracy = (
        correct / max(actual, detected)
    ) * 100

    accuracy_list.append(accuracy)

    # ================= SAVE RESULT =================

    result = {

        "Video": video["name"],

        "Actual": actual,

        "Detected": detected,

        "False": false_detections,

        "Correct": correct,

        "Accuracy": round(accuracy, 2)
    }

    all_results.append(result)

    # ================= TERMINAL OUTPUT =================

    print("\n========== RESULT ==========")

    print(f"Video                : {video['name']}")
    print(f"Actual Potholes      : {actual}")
    print(f"Detected Potholes    : {detected}")
    print(f"False Detections     : {false_detections}")
    print(f"Correct Detections   : {correct}")
    print(f"Accuracy             : {round(accuracy, 2)}%")

    print("============================")

    cap.release()
    cv.destroyAllWindows()

# ================= FINAL OVERALL ACCURACY =================

final_accuracy = sum(accuracy_list) / len(accuracy_list)

print("\n====================================")
print("FINAL OVERALL PERFORMANCE")
print("====================================")

print(f"Final Overall Accuracy : {round(final_accuracy, 2)}%")

print("====================================")

# ================= SAVE CSV =================

with open("final_results.csv", "w", newline="") as csvfile:

    fieldnames = [

        "Video",
        "Actual",
        "Detected",
        "False",
        "Correct",
        "Accuracy"
    ]

    writer = csv.DictWriter(
        csvfile,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for row in all_results:
        writer.writerow(row)

print("\nResults saved to final_results.csv")