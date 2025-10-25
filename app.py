
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load models
models = {
    "recommended_product": joblib.load("recommended_product_model.pkl"),
    "home_remedy": joblib.load("home_remedy_model.joblib"),
    "stress_level": joblib.load("stress_level_model.pkl"),
    "exercise_recommendation": joblib.load("exercise_recommendation_model.joblib"),
    "exercise_duration": joblib.load("exercise_duration_model.joblib"),
    "exercise_description": joblib.load("exercise_description_model.joblib"),
    "foods_to_eat": joblib.load("foods_to_eat_model.joblib"),
    "foods_to_avoid": joblib.load("foods_to_avoid_model.joblib")
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_product', methods=['GET', 'POST'])
def predict_product():
    if request.method == 'POST':
        age = int(request.form['Age'])
        gender = request.form['Gender']
        skin_type = request.form['Skin_Type']
        skin_tone = request.form['Skin_Tone']
        skin_concern = request.form['Skin_Concern']

        input_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Skin_Type': skin_type,
            'Skin_Tone': skin_tone,
            'Skin_Concern': skin_concern
        }])

        model = models['recommended_product']
        result = model.predict(input_data)[0]

        return render_template('product_result.html', prediction=result)

    # If method is GET (form load), show the form
    return render_template('predict_product.html')




@app.route('/predict_remedy', methods=['GET'])
def predict_remedy():
    return render_template('predict_remedy.html')

@app.route('/result_remedy', methods=['POST'])
def result_remedy():
    concern = request.form['Concern']

    model = models['home_remedy']
    input_df = pd.DataFrame({'Skin_Concern': [concern]})  # ✅ Match training column name
    prediction = model.predict(input_df)[0]

    return render_template('remedy_result.html', result=prediction, concern=concern)



@app.route('/predict_stress', methods=['GET', 'POST'])
def predict_stress():
    if request.method == 'POST':
        data = [[
            float(request.form['Sleep_Hours']),
            float(request.form['Screen_Time'])
        ]]
        result = models['stress_level'].predict(data)[0]
        return render_template('stress_result.html', result=result)
    return render_template('predict_stress.html')

@app.route('/predict_exercise', methods=['GET', 'POST'])
def predict_exercise():
    if request.method == 'POST':
        sleep = float(request.form['Sleep_Hours'])
        screen = float(request.form['Screen_Time'])

        input_data = pd.DataFrame([{
            'Sleep_Hours': sleep,
            'Screen_Time_Hours': screen  # ✅ match training column
        }])

        exercise = models['exercise_recommendation'].predict(input_data)[0]
        duration = models['exercise_duration'].predict(input_data)[0]
        description = models['exercise_description'].predict(input_data)[0]

        return render_template('exercise_result.html',
                               exercise=exercise,
                               duration=duration,
                               description=description)

    return render_template('predict_exercise.html')


# @app.route('/predict_foods', methods=['GET', 'POST'])
# def predict_foods():
#     if request.method == 'POST':
#         data = [[request.form['Skin_Concern'], request.form['Diet_Type']]]
#         eat = models['foods_to_eat'].predict(data)[0]
#         avoid = models['foods_to_avoid'].predict(data)[0]
#         return render_template('foods_result.html', eat=eat, avoid=avoid)
#     return render_template('predict_foods.html')

if __name__ == '__main__':
    app.run(debug=True)  