import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

data = pd.read_csv("student-por.csv", delimiter=";")

study_hours = data["studytime"]
exam_scores = (data["G1"] + data["G2"] + data["G3"]) / 3

average_scores = {}
for hours, score in zip(study_hours, exam_scores):
    if hours not in average_scores:
        average_scores[hours] = [score]
    else:
        average_scores[hours].append(score)

study_hours_average = []
scores_average = []

for hours, scores in average_scores.items():
    study_hours_average.append(hours)
    scores_average.append(sum(scores) / len(scores))

study_hours_average = np.array(study_hours_average).reshape(-1, 1)
scores_average = np.array(scores_average).reshape(-1, 1)

marks_predictor = LinearRegression()
marks_predictor.fit(study_hours_average, scores_average)

what_marks = marks_predictor.predict([[0.3]])
print(what_marks)

plt.scatter(study_hours_average, scores_average)
plt.plot(study_hours_average, marks_predictor.predict(study_hours_average), color='red', linewidth=2)
plt.xlabel("Study Hours")
plt.ylabel("Average Exam Scores")
plt.title("Average Exam Scores vs. Study Hours")
plt.show()
