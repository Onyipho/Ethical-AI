# Ethical-AI
In the realm of machine learning, the pervasive issue of bias within datasets and models can inadvertently lead to unfair and discriminatory outcomes. This project introduces a web-based tool (prototyped in with streamlit) designed to detect and mitigate biases present in datasets and predictive models. Utilizing established datasets such as the UCI Adult Income and German Credit datasets, our tool performs a comprehensive analysis to identify potential biases across different demographic groups. The tool employs logistic regression to evaluate model performance, assessing metrics such as accuracy, precision, and recall for different subgroups.
A key feature of the tool is its ability to suggest targeted mitigation strategies based on the detected biases, including techniques such as data balancing, implementing fairness constraints, and adjusting decision thresholds. Currently, the tool is designed to handled binary classification models. It will be upgraded to a more robust system in the future.
