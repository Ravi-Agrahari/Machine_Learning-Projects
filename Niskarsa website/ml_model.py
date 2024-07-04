import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def clean_data(data):
    """
    Clean the data by converting all columns to numeric, forcing errors to NaN,
    and then filling NaN values with the column mean.
    """
    data = data.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric, coerce errors to NaN
    data = data.fillna(data.mean())  # Fill NaN values with column means
    return data

def analyze_and_predict(data, subjects=['English', 'Science', 'Maths', 'History', 'Geography']):
    # Clean the data
    data = clean_data(data)
    
    # Filter data for specified subjects
    data = data[subjects]
    
    # Calculate mean for each subject
    subject_means = data.mean().round(2)  # Round to two decimal places

    # Calculate total average
    total_average = data.mean().mean().round(2)  # Round to two decimal places

    # Add total average to the subject means dictionary
    subject_means['Total'] = total_average

    suggestions = {}

    # Generate suggestions based on mean scores
    for subject in subjects:
        if subject_means[subject] < 60:
            suggestions[subject] = "Need to improve"
        else:
            suggestions[subject] = "Doing well"

    # Ensure the plots directory exists
    if not os.path.exists('static/plots'):
        os.makedirs('static/plots')

    # Plotting histograms for each subject
    plot_paths = []
    for subject in subjects:
        plt.figure()
        sns.histplot(data[subject], kde=True)
        plt.title(f'Distribution of {subject}')
        plot_path = f'static/plots/{subject}.png'
        plt.savefig(plot_path)
        plot_paths.append(plot_path)
        plt.close()

    return subject_means, suggestions, plot_paths


def predict_next_semester(data, subjects=['English', 'Science', 'Maths', 'History', 'Geography']):
    # Clean the data
    data = clean_data(data)

    # Filter data for specified subjects
    data = data[subjects]

    # Simple prediction example: next semester's grades
    predictions = (data.mean() * np.random.uniform(0.9, 1.1)).round(2)  # Round to two decimal places

    # Calculate total
    total_prediction = predictions.sum().round(2)

    # Add total prediction to the predictions dictionary
    predictions['Total'] = total_prediction

    return predictions

