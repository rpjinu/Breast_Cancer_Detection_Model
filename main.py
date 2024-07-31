import pickle
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

# Load the model and scaler from pickle files
with open('best_svm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


# Function to make predictions
def predict_breast_cancer(input_data):
    # Convert input data to numpy array and reshape
    input_array = np.array(input_data).reshape(1, -1)

    # Apply scaling
    scaled_input = scaler.transform(input_array)

    # Make prediction
    prediction = model.predict(scaled_input)
    return prediction[0]


# Main function to take user input and make prediction
def main():
    print("Enter the following details:")
    sample_code_number = int(input("Sample code number: "))
    clump_thickness = int(input("Clump Thickness (1-10): "))
    uniformity_cell_size = int(input("Uniformity of Cell Size (1-10): "))
    uniformity_cell_shape = int(input("Uniformity of Cell Shape (1-10): "))
    marginal_adhesion = int(input("Marginal Adhesion (1-10): "))
    single_epithelial_cell_size = int(input("Single Epithelial Cell Size (1-10): "))
    bare_nuclei = int(input("Bare Nuclei (1-10): "))
    bland_chromatin = int(input("Bland Chromatin (1-10): "))
    normal_nucleoli = int(input("Normal Nucleoli (1-10): "))
    mitoses = int(input("Mitoses (1-10): "))

    # Collect input data into a list
    input_data = [
        sample_code_number,
        clump_thickness,
        uniformity_cell_size,
        uniformity_cell_shape,
        marginal_adhesion,
        single_epithelial_cell_size,
        bare_nuclei,
        bland_chromatin,
        normal_nucleoli,
        mitoses
    ]

    # Make a prediction
    result = predict_breast_cancer(input_data)

    # Output the result
    print(f"The model prediction is: {'Malignant' if result == 1 else 'Benign'}")


if __name__ == "__main__":
    main()
