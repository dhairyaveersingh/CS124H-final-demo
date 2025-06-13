

---

# Emotion Classifier App: Veer

## Overview
The **Emotion Classifier App** is a Streamlit-based web application that uses a pre-trained transformer model to analyze the emotions expressed in text. It provides sentiment classification with probability scores and enables monitoring of user interactions.

## Features
- **Emotion Classification**: Uses the `j-hartmann/emotion-english-roberta-large` model to predict emotions.
- **Interactive Interface**: Built with Streamlit for an easy-to-use web experience.
- **Visualization**: Displays sentiment probabilities with **Altair** and **Plotly** charts.
- **User Monitoring**: Tracks page visits and prediction details for deeper insights.

## Technologies Used
- `streamlit`
- `altair`
- `plotly.express`
- `pandas`, `numpy`
- `transformers`
- `datetime`
- Custom utility functions (`track_utils`)

## Installation
Ensure you have Python installed, then install the required dependencies:

```bash
pip install streamlit altair plotly pandas numpy transformers
```

## Usage
Run the application with:

```bash
streamlit run app.py
```

Navigate through the sidebar menu:
- 🏠 **Home**: Enter text to analyze its emotion.
- 📊 **Monitor**: View app metrics, including page visits and emotion predictions.
- ℹ️ **About**: Learn about the app and its purpose.

## Emotion Categories
The app supports the following emotions, displayed with respective emoji icons:
- **Anger** 😠
- **Disgust** 🤮
- **Fear** 😨😱
- **Happiness** 🤗
- **Joy** 😂
- **Neutral** 😐
- **Sadness** 😔
- **Surprise** 😮

## Custom Styling
The app includes custom CSS to enhance the UI, modifying sidebar colors, button styles, and data visualization aesthetics.

## Contributing
Feel free to improve the app by submitting a pull request! Any ideas for enhanced visualization or better emotion modeling are welcome.

## License
This project is licensed under the MIT License.

---


