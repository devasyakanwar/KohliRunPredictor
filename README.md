# RunsPrediction
Predicting Runs made by Virat Kohli using nueral networks with three layers two running on linear regression and last one on ReLU and predicts run based on opponent using dataset


Virat Kohli Runs Predictor
This project is a machine learning web application that predicts the runs Virat Kohli might score in a cricket match given various match conditions and his recent performance. It uses:

PyTorch for building and training a simple neural network.
Streamlit for creating a user-friendly web interface.
Matplotlib for visualizing the training loss curve.
scikit-learn for label-encoding categorical features.
Table of Contents
Features
Tech Stack
Project Structure
Installation
Usage
Model Details
Screenshot
Contributing
License
Features
User Inputs: Enter match details such as Opponent, Match Format, Venue, Pitch Type, Weather, Toss Decision, Bowler Type Faced Most, and Kohli’s last 5 scores.
Real-Time Predictions: Click a button to see how many runs Kohli might score based on these conditions.
Training Loss Visualization: Displays a chart of the model’s training loss over epochs.
Dark-Themed UI: The app uses a dark theme for a modern, sleek look.
Tech Stack
Python 3.7+
PyTorch for the model (torch, torch.nn, torch.optim)
Streamlit for the web interface
scikit-learn for label encoding
Matplotlib for plotting the training loss
Project Structure
arduino
Copy
.
├── app.py
├── .streamlit
│   └── config.toml
├── README.md
└── requirements.txt (optional)
app.py: Main Streamlit application, including data preprocessing, model definition, training, and UI code.
.streamlit/config.toml: Configuration file for Streamlit’s dark theme.
README.md: Project documentation (this file).
requirements.txt: List of Python dependencies (optional).
Installation
Clone or Download this repository.

Create a Virtual Environment (recommended but optional):

bash
Copy
python -m venv venv
# Activate it:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
Install Dependencies:

bash
Copy
pip install streamlit torch torchvision torchaudio scikit-learn matplotlib
Or, if you prefer conda:

bash
Copy
conda install pytorch scikit-learn matplotlib -c pytorch
pip install streamlit
Set up Dark Theme:

Make sure you have a .streamlit folder in the same directory as app.py.
Inside .streamlit, create a config.toml with the following content:
toml
Copy
[theme]
base="dark"
primaryColor="#F63366"
backgroundColor="#0E1117"
secondaryBackgroundColor="#262730"
textColor="#FFFFFF"
This ensures Streamlit uses the dark theme.
Usage
Open a Terminal in the project folder (where app.py is located).
Run the Streamlit App:
bash
Copy
streamlit run app.py
or
bash
Copy
python -m streamlit run app.py
Open the Local URL (usually http://localhost:8501) in your browser to access the app.
Interacting with the App
Match Conditions: Use the top row of dropdowns/radio buttons to select Opponent, Match Format, Venue, Pitch Type, Weather, Toss Decision, and Bowler Type.
Kohli’s Last 5 Scores: Enter the last 5 scores in the provided numeric fields.
Predict: Click the Predict Runs button to see the model’s prediction.
Training Loss Curve: Scroll down to see how the training loss evolved over 300 epochs.
Model Details
Model Architecture:
A simple feed-forward neural network:
Input Layer: Size = number of features (7 categorical + 5 numeric scores = 12).
Hidden Layer: 16 units, ReLU activation.
Output Layer: 1 unit (predicted runs).
Loss Function: Mean Squared Error (MSE).
Optimizer: Adam, learning rate = 0.01.
Training Data: A small mock dataset with 10 rows for demonstration. Each row includes the categorical columns, last 5 scores, and a Predicted_Runs label for supervised training.
Screenshot
Below is an example screenshot of the dark-themed UI with horizontally arranged inputs at the top and a training loss curve at the bottom.


(Replace the URL with an actual screenshot if you want to show the real UI.)

Contributing
Contributions are welcome! If you’d like to:

Fix a bug
Add new features
Refactor code
Please open a pull request or raise an issue.

License
This project is open-source. You can include a license of your choice (e.g., MIT, Apache, etc.). For example:

sql
Copy
MIT License

Copyright (c) 2025 ...

Permission is hereby granted, free of charge, to any person obtaining a copy
...
That’s it! Feel free to update this README with more details about your project, references, or screenshots as you see fit.





