import pandas as pd
import matplotlib.pyplot as plt

# ================= READ CSV =================

df = pd.read_csv("final_results.csv")

print("\n========== VIDEO RESULTS ==========\n")

print(df)

# ================= FINAL ACCURACY =================

total_actual = df["Actual"].sum()

total_correct = df["Correct"].sum()

# FINAL ACCURACY = MEAN OF ALL VIDEO ACCURACIES
final_accuracy = df["Accuracy"].mean()

print("\n===================================")
print("FINAL PERFORMANCE ANALYSIS")
print("===================================")

print(f"Total Actual Potholes    : {total_actual}")

print(f"Total Correct Detections : {total_correct}")

print(f"Final Overall Accuracy   : {round(final_accuracy, 2)}%")

print("===================================")

# ================= BAR GRAPH =================

videos = df["Video"]

accuracies = df["Accuracy"]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    videos,
    accuracies,
    color=['blue', 'green', 'orange', 'red']
)

plt.title("Video-wise Detection Accuracy")

plt.xlabel("Videos")

plt.ylabel("Accuracy (%)")

plt.ylim(0, 100)

# Display values on bars

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 1,
        f'{round(height,1)}%',
        ha='center'
    )

plt.show()

# ================= ACTUAL VS DETECTED =================

actual = df["Actual"]

detected = df["Detected"]

x = range(len(videos))

plt.figure(figsize=(8, 5))

plt.bar(
    x,
    actual,
    width=0.4,
    label="Actual"
)

plt.bar(
    [i + 0.4 for i in x],
    detected,
    width=0.4,
    label="Detected"
)

plt.xticks(
    [i + 0.2 for i in x],
    videos
)

plt.xlabel("Videos")

plt.ylabel("Pothole Count")

plt.title("Actual vs Detected Potholes")

plt.legend()

plt.show()

# ================= PIE CHART =================

total_false = df["False"].sum()

sizes = [total_correct, total_false]

labels = ["Correct Detections", "False Detections"]

colors = ["green", "red"]

plt.figure(figsize=(6, 6))

plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    colors=colors
)

plt.title("Detection Performance Distribution")

plt.show()

# ================= SAVE SUMMARY =================

summary = f"""
========== FINAL ANALYSIS ==========

Total Actual Potholes    : {total_actual}

Total Correct Detections : {total_correct}

Final Overall Accuracy   : {round(final_accuracy, 2)}%

===================================
"""

with open("final_analysis.txt", "w") as f:

    f.write(summary)

print("\nAnalysis saved to final_analysis.txt")