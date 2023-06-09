import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("student-por.csv")

study_hours = data["study_hours"]
exam_scores = data["exam_scores"]


plt.scatter(study_hours, exam_scores)
plt.xlabel("Study Hours")
plt.ylabel("Exam Scores")
plt.title("Student Exam Scores vs. Study Hours")
plt.show()
