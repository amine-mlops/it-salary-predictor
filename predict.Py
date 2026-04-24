import numpy as np

def predict_salary(features, w, b):
    return np.dot(features, w) + b

def main():
    w_final = np.array([1.1047156, 0.15053473, 0.09541467, 0.85183099])
    b_final = 6.754835710004856

    print("=== Salary prediction tool (K MAD) ===")

    try:
        exp = float(input("Years of Experience: "))
        yic = float(input("Years in Company: "))
        cert = float(input("Certifications: "))
        proj = float(input("Projects Completed: "))

        user_data = np.array([exp, yic, cert, proj])
        
        res = predict_salary(user_data, w_final, b_final)

        print("-" * 30)
        print(f"Estimated Salary: {res:.2f} k MAD")
        print("-" * 30)
    except ValueError:
        print("Error: Please enter numeric values.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPrediction stopped by user. Goodbye!")