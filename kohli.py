import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# 1) SAMPLE DATA (Including Kohli_Last_5_Scores as lists)

data = {
    'Opponent': [
        "Australia", "Pakistan", "England", "New Zealand", "South Africa", 
        "Sri Lanka", "West Indies", "Bangladesh", "Afghanistan", "India A"
    ],
    'Match_Format': [
        "ODI", "T20", "Test", "ODI", "T20", 
        "Test", "ODI", "T20", "Test", "ODI"
    ],
    'Venue': [
        "Home", "Away", "Neutral", "Away", "Home", 
        "Neutral", "Home", "Away", "Neutral", "Home"
    ],
    'Pitch_Type': [
        "Batting-friendly", "Bowling-friendly", "Balanced", "Batting-friendly", "Balanced", 
        "Bowling-friendly", "Batting-friendly", "Bowling-friendly", "Balanced", "Batting-friendly"
    ],
    'Weather': [
        "Sunny", "Overcast", "Humid", "Sunny", "Overcast", 
        "Humid", "Sunny", "Overcast", "Humid", "Sunny"
    ],
    'Toss_Decision': [
        "Bat", "Bowl", "Bat", "Bat", "Bowl", 
        "Bowl", "Bat", "Bowl", "Bat", "Bat"
    ],
    'Bowler_Type_Faced_Most': [
        "Fast", "Spin", "Fast", "Spin", "Fast", 
        "Spin", "Fast", "Spin", "Fast", "Spin"
    ],
    'Kohli_Last_5_Scores': [
        [45, 78, 33, 89, 20],
        [10, 23, 55, 47, 60],
        [101, 77, 54, 120, 98],
        [5, 34, 78, 88, 90],
        [76, 22, 45, 33, 100],
        [120, 98, 140, 23, 77],
        [15, 55, 62, 75, 81],
        [30, 40, 50, 60, 70],
        [80, 90, 100, 110, 120],
        [25, 35, 45, 55, 65]
    ],
    'Predicted_Runs': [85, 45, 110, 67, 80, 130, 55, 40, 120, 60]
}

df = pd.DataFrame(data)


# 2) EXPAND 'Kohli_Last_5_Scores' INTO 5 SEPARATE COLUMNS

df[['Score1','Score2','Score3','Score4','Score5']] = pd.DataFrame(
    df['Kohli_Last_5_Scores'].tolist(),
    index=df.index
)

# Drop the original list column
df.drop('Kohli_Last_5_Scores', axis=1, inplace=True)


# 3) LABEL-ENCODE CATEGORICAL COLUMNS

categorical_cols = [
    'Opponent', 'Match_Format', 'Venue', 
    'Pitch_Type', 'Weather', 'Toss_Decision', 
    'Bowler_Type_Faced_Most'
]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# 4) PREPARE FEATURES (X) & TARGET (y)

X = torch.tensor(df.drop('Predicted_Runs', axis=1).values, dtype=torch.float32)
y = torch.tensor(df['Predicted_Runs'].values, dtype=torch.float32)


# 5) DEFINE & TRAIN MODEL

class KohliRunPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = KohliRunPredictor(input_size=X.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 300
loss_values = []
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X).squeeze()
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    loss_values.append(loss.item())

# 6) STREAMLIT APP (UI DESIGN SIMILAR TO YOUR SCREENSHOT)

st.set_page_config(page_title="Kohli Runs Predictor", layout="wide")

# Main Title
st.markdown("<h1 style='text-align: center;'>Virat Kohli Runs Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# 6a) TOP ROW OF INPUTS (CATEGORICAL)
top_cols = st.columns([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
with top_cols[0]:
    opponent = st.selectbox("Opponent", label_encoders['Opponent'].classes_)
with top_cols[1]:
    match_format = st.selectbox("Match Format", label_encoders['Match_Format'].classes_)
with top_cols[2]:
    venue = st.selectbox("Venue", label_encoders['Venue'].classes_)
with top_cols[3]:
    pitch_type = st.selectbox("Pitch Type", label_encoders['Pitch_Type'].classes_)
with top_cols[4]:
    weather = st.selectbox("Weather", label_encoders['Weather'].classes_)
with top_cols[5]:
    toss_decision = st.radio("Toss Decision", label_encoders['Toss_Decision'].classes_)
with top_cols[6]:
    bowler_type = st.radio("Bowler Type", label_encoders['Bowler_Type_Faced_Most'].classes_)

# 6b) SECOND ROW OF INPUTS (LAST 5 SCORES) + PREDICT BUTTON
bot_cols = st.columns([1,1,1,1,1,1])
last_5_scores = []
for i in range(5):
    with bot_cols[i]:
        val = st.number_input(f"Score {i+1}", min_value=0, max_value=300, value=50)
        last_5_scores.append(val)

predict_button_col = bot_cols[5]
with predict_button_col:
    st.markdown("<br/>", unsafe_allow_html=True)  
    if st.button("Predict Runs"):
        opp_val = label_encoders['Opponent'].transform([opponent])[0]
        mf_val = label_encoders['Match_Format'].transform([match_format])[0]
        ven_val = label_encoders['Venue'].transform([venue])[0]
        pt_val = label_encoders['Pitch_Type'].transform([pitch_type])[0]
        weath_val = label_encoders['Weather'].transform([weather])[0]
        toss_val = label_encoders['Toss_Decision'].transform([toss_decision])[0]
        bowler_val = label_encoders['Bowler_Type_Faced_Most'].transform([bowler_type])[0]

        input_features = torch.tensor([
            opp_val, mf_val, ven_val, pt_val, weath_val,
            toss_val, bowler_val,
            *last_5_scores
        ], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            predicted_runs = model(input_features).item()

        st.success(f"Predicted Runs: {predicted_runs:.2f}")

st.markdown("---")

# 6c)TRAINING LOSS CURVE
st.error("Slow Training Loss Curve (just like real-time model training!)")
fig, ax = plt.subplots()
ax.plot(range(epochs), loss_values, color='cyan', label='Loss')
ax.set_xlabel("Epochs")
ax.set_ylabel("MSE Loss")
ax.set_title("Training Loss Curve")
ax.legend()
st.pyplot(fig)
