import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('Model/meal_recommendation_model.pkl')
gender_le = joblib.load('Model/gender_le.pkl')
goal_le = joblib.load('Model/goal_le.pkl')
meal_le = joblib.load('Model/meal_le.pkl')

# Load meals dataset
meals_df = pd.read_csv('Dataset/indian_meals_1000_dataset.csv') 

# BMR Calculation without activity level input
def calculate_bmr(row):
    if row['Gender'] == gender_le.transform(['male'])[0]:
        bmr = 10 * row['Weight_kg'] + 6.25 * row['Height_cm'] - 5 * row['Age'] + 5
    else:
        bmr = 10 * row['Weight_kg'] + 6.25 * row['Height_cm'] - 5 * row['Age'] - 161

    # Default activity multiplier (lightly active)
    maintenance = bmr * 1.375

    if row['Goal'] == goal_le.transform(['weight_loss'])[0]:
        return maintenance - 400
    elif row['Goal'] == goal_le.transform(['weight_gain'])[0]:
        return maintenance + 400
    return maintenance

# Prediction function (no activity input)
def predict_meal(age, gender, height, weight, goal):
    gender_enc = gender_le.transform([gender])[0]
    goal_enc = goal_le.transform([goal])[0]

    bmr_user = calculate_bmr({
        "Age": age,
        "Gender": gender_enc,
        "Height_cm": height,
        "Weight_kg": weight,
        "Goal": goal_enc
    })

    input_vec = pd.DataFrame(
        [[age, gender_enc, height, weight, goal_enc]],
        columns=['Age', 'Gender', 'Height_cm', 'Weight_kg', 'Goal']
    )

    prediction = model.predict(input_vec)[0]

    return {
        "Breakfast": meal_le.inverse_transform([prediction[0]])[0],
        "Lunch": meal_le.inverse_transform([prediction[1]])[0],
        "Dinner": meal_le.inverse_transform([prediction[2]])[0]
    }

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🍽️ Personalized Meal Recommendation")

# Input fields (no activity)
age = st.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.selectbox("Gender", gender_le.classes_)
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
goal = st.selectbox("Fitness Goal", goal_le.classes_)

if st.button("Get Meal Plan"):
    try:
        result = predict_meal(age, gender, height, weight, goal)

        st.subheader("📋 Meal Details (Name, Calories, Protein, Carbs, Fat)")
        for meal_type, meal_code in result.items():
            meal_info = meals_df[meals_df['Meal_ID'] == meal_code]
            if not meal_info.empty:
                display_info = meal_info[['Name', 'Calories', 'Protein', 'Carbs', 'Fat']]
                st.markdown(f"### {meal_type}")
                st.dataframe(display_info)
            else:
                st.warning(f"No details found for meal code: {meal_code}")

    except ValueError as e:
        st.error(f"Error: {e}")
