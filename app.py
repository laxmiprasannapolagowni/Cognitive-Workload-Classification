import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load Voting Models (.sav)
wafd_binary = joblib.load("models/wafd_binary.sav")
wafd_multi = joblib.load("models/wafd_multi.sav")
afd_binary = joblib.load("models/afd_binary.sav")
afd_multi = joblib.load("models/afd_multi.sav")

scaler_wafd = joblib.load("models/scaler_wafd.pkl")
scaler_afd = joblib.load("models/scaler_afd.pkl")

binary_map = {0: "Low Load", 1: "High Load"}
multi_map = {0: "Low", 1: "Moderate", 2: "High"}

wafd_features = [
    'HR_mean','HR_std','RR_mean','RR_rmssd','GSR_mean','GSR_std',
    'TEMP_mean','ACC_mean','ACC_std','Tap_speed'
]

afd_features = [
    'HR_mean','HR_std','RR_mean','RR_rmssd','GSR_mean','GSR_std',
    'TEMP_mean','ACC_mean','ACC_std','Tap_speed','HR_GSR',
    'RR_HR_ratio','ACC_TEMP','HR_sq','GSR_sq','PCA_1','PCA_2'
]

def get_tips(level):
    if "High" in level:
        return [
            "Practice 4-7-8 breathing technique",
            "Take a 5–10 minute mental break",
            "Reduce multitasking",
            "Stay hydrated",
            "Perform short physical stretching"
        ]
    elif "Moderate" in level:
        return [
            "Maintain steady breathing",
            "Take micro breaks",
            "Prioritize tasks",
            "Avoid unnecessary distractions"
        ]
    else:
        return [
            "Keep maintaining your current pace",
            "Stay consistent with healthy habits",
            "Continue monitoring workload"
        ]

@app.route("/")
def home():
    return render_template("index.html", features=afd_features)

@app.route("/predict", methods=["POST"])
def predict():

    inputs = [float(request.form.get(f)) for f in afd_features]

    afd_input = np.array(inputs).reshape(1,-1)
    wafd_input = np.array(inputs[:10]).reshape(1,-1)

    afd_scaled = scaler_afd.transform(afd_input)
    wafd_scaled = scaler_wafd.transform(wafd_input)

    # Predictions
    wafd_bin_pred = wafd_binary.predict(wafd_scaled)[0]
    wafd_multi_pred = wafd_multi.predict(wafd_scaled)[0]

    afd_bin_pred = afd_binary.predict(afd_scaled)[0]
    afd_multi_pred = afd_multi.predict(afd_scaled)[0]

    # Soft Voting Confidence
    wafd_bin_conf = np.max(wafd_binary.predict_proba(wafd_scaled))*100
    wafd_multi_conf = np.max(wafd_multi.predict_proba(wafd_scaled))*100
    afd_bin_conf = np.max(afd_binary.predict_proba(afd_scaled))*100
    afd_multi_conf = np.max(afd_multi.predict_proba(afd_scaled))*100

    # Feature Importance (try extracting from LightGBM inside Voting)
    importance = np.zeros(len(afd_features))
    try:
        lgbm_model = afd_binary.named_estimators_['LightGBM']
        importance = lgbm_model.feature_importances_
    except:
        pass

    # Tips based on AFD Multi prediction
    tips = get_tips(multi_map[afd_multi_pred])

    return render_template("result.html",
                           wafd_bin=binary_map[wafd_bin_pred],
                           wafd_multi=multi_map[wafd_multi_pred],
                           afd_bin=binary_map[afd_bin_pred],
                           afd_multi=multi_map[afd_multi_pred],
                           wafd_bin_conf=round(wafd_bin_conf,2),
                           wafd_multi_conf=round(wafd_multi_conf,2),
                           afd_bin_conf=round(afd_bin_conf,2),
                           afd_multi_conf=round(afd_multi_conf,2),
                           importance=importance.tolist(),
                           feature_names=afd_features,
                           tips=tips)

if __name__ == "__main__":
    app.run()
