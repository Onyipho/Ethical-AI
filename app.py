import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Function to preprocess the data
def preprocess_data(data):
    data.dropna(inplace=True)
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
    return data, label_encoders

# Function to split the data
def split_data(data, target_column, protected_attribute_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    protected_attribute = data[protected_attribute_column]
    X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
        X, y, protected_attribute, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, p_train, p_test

# Function to train the model
def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, multi_class='ovr')
    model.fit(X_train_scaled, y_train)
    return model, scaler

# Function to evaluate the model
def evaluate_model(model, scaler, X_test, y_test, p_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Check if y_pred_proba is a 1D array or 2D array
    if y_pred_proba.ndim == 2:
        y_pred_proba = y_pred_proba[:, 1]
    else:
        y_pred_proba = y_pred_proba
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    metrics, disparity = calculate_disparity(y_pred, y_test, p_test)
    
    return accuracy, conf_matrix, roc_auc, metrics, disparity

# Function to calculate disparity
def calculate_disparity(preds, labels, protected_attr):
    outcomes = pd.DataFrame({'pred': preds, 'label': labels, 'protected': protected_attr})
    group_0 = outcomes[outcomes['protected'] == 0]
    group_1 = outcomes[outcomes['protected'] == 1]
    
    metrics = {}
    for group, name in [(group_0, 'Group 0'), (group_1, 'Group 1')]:
        accuracy = accuracy_score(group['label'], group['pred'])
        precision = confusion_matrix(group['label'], group['pred'])[1, 1] / (confusion_matrix(group['label'], group['pred'])[1, 1] + confusion_matrix(group['label'], group['pred'])[0, 1])
        recall = confusion_matrix(group['label'], group['pred'])[1, 1] / (confusion_matrix(group['label'], group['pred'])[1, 1] + confusion_matrix(group['label'], group['pred'])[1, 0])
        metrics[name] = {'accuracy': accuracy, 'precision': precision, 'recall': recall}
    
    disparity = {
        'accuracy': metrics['Group 1']['accuracy'] / metrics['Group 0']['accuracy'],
        'precision': metrics['Group 1']['precision'] / metrics['Group 0']['precision'],
        'recall': metrics['Group 1']['recall'] / metrics['Group 0']['recall'],
    }
    
    return metrics, disparity

# Function to suggest mitigation strategies
def suggest_mitigation_strategies(disparity):
    suggestions = []
    
    if disparity['accuracy'] < 0.9:
        suggestions.append("Consider balancing the dataset using oversampling or undersampling techniques.")
    
    if disparity['precision'] > 1.2:
        suggestions.append("Implement fairness constraints in the model training process to ensure equal precision.")
    
    if disparity['recall'] > 2.0:
        suggestions.append("Use post-processing techniques like adjusting decision thresholds to equalize recall rates.")
    
    return suggestions

# Function to plot the metrics
def plot_metrics(metrics):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i, metric in enumerate(['accuracy', 'precision', 'recall']):
        sns.barplot(x=['Group 0', 'Group 1'], y=[metrics['Group 0'][metric], metrics['Group 1'][metric]], ax=ax[i])
        ax[i].set_title(f'{metric.capitalize()} by Group')
    st.pyplot(fig)

# Streamlit app layout
st.title("Bias Detection and Mitigation Tool")
st.write("Upload a dataset to analyze for bias and suggest mitigation strategies.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(data.head())

    data, label_encoders = preprocess_data(data)
    
    target_column = st.selectbox("Select the target column", data.columns)
    protected_attribute_column = st.selectbox("Select the protected attribute column", data.columns)
    
    if st.button("Analyze"):
        X_train, X_test, y_train, y_test, p_train, p_test = split_data(data, target_column, protected_attribute_column)
        model, scaler = train_model(X_train, y_train)
        
        accuracy, conf_matrix, roc_auc, metrics, disparity = evaluate_model(model, scaler, X_test, y_test, p_test)
        
        st.write(f'Overall Accuracy: {accuracy}')
        st.write(f'Confusion Matrix:\n{conf_matrix}')
        st.write(f'ROC AUC Score: {roc_auc}')
        
        # Display Metrics by Group in a Table
        st.markdown("### Metrics by Group:")
        metrics_data = {
            "Metric": ["Accuracy", "Precision", "Recall"],
            "Group 0": [metrics['Group 0']['accuracy'], metrics['Group 0']['precision'], metrics['Group 0']['recall']],
            "Group 1": [metrics['Group 1']['accuracy'], metrics['Group 1']['precision'], metrics['Group 1']['recall']]
        }
        metrics_df = pd.DataFrame(metrics_data)
        st.table(metrics_df)

        # Display Disparity in a Table
        st.markdown("### Disparity:")
        disparity_data = {
            "Metric": ["Accuracy", "Precision", "Recall"],
            "Disparity": [disparity['accuracy'], disparity['precision'], disparity['recall']]
        }
        disparity_df = pd.DataFrame(disparity_data)
        st.table(disparity_df)

        # Detailed Explanation
        st.markdown("""
        ### Detailed Explanation:
        1. **Metrics by Group**:
           - **Group 0**:
             - **Accuracy**: {:.3f} (or {:.1f}%)
             - **Precision**: {:.3f} (or {:.1f}%)
             - **Recall**: {:.3f} (or {:.1f}%)
           - **Group 1**:
             - **Accuracy**: {:.3f} (or {:.1f}%)
             - **Precision**: {:.3f} (or {:.1f}%)
             - **Recall**: {:.3f} (or {:.1f}%)
        
        2. **Disparity**:
           - **Accuracy** Disparity: {:.3f} (or {:.1f}%)
             - This means the accuracy for Group 1 is {:.1f}% of the accuracy for Group 0.
           - **Precision** Disparity: {:.3f} (or {:.1f}%)
             - This indicates that the precision for Group 1 is {:.1f}% of the precision for Group 0.
           - **Recall** Disparity: {:.3f} (or {:.1f}%)
             - This indicates that the recall for Group 1 is {:.1f}% of the recall for Group 0.
        """.format(
            metrics['Group 0']['accuracy'], metrics['Group 0']['accuracy']*100,
            metrics['Group 0']['precision'], metrics['Group 0']['precision']*100,
            metrics['Group 0']['recall'], metrics['Group 0']['recall']*100,
            metrics['Group 1']['accuracy'], metrics['Group 1']['accuracy']*100,
            metrics['Group 1']['precision'], metrics['Group 1']['precision']*100,
            metrics['Group 1']['recall'], metrics['Group 1']['recall']*100,
            disparity['accuracy'], disparity['accuracy']*100,
            disparity['accuracy']*100,
            disparity['precision'], disparity['precision']*100,
            disparity['precision']*100,
            disparity['recall'], disparity['recall']*100,
            disparity['recall']*100
        ))
             
        plot_metrics(metrics)
        
        suggestions = suggest_mitigation_strategies(disparity)
        st.write("Suggested Mitigation Strategies:")
        for suggestion in suggestions:
            st.write(f'- {suggestion}')


 