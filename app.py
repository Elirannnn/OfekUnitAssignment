from flask import Flask, render_template, request, jsonify, send_file, url_for
from flask_cors import CORS
import os
import pandas as pd
import re
import tempfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

app = Flask(__name__)
CORS(app)

def create_spelling_model():
    # Known correct unit names
    correct_units = ['הנדסה', 'מאב', 'מהן', 'מעמ', 'מתן', 'מטס', 'תשתיות']
    
    # Generate training data with common misspellings
    training_data = []
    labels = []
    
    # Add correct spellings
    for unit in correct_units:
        training_data.append(unit)
        labels.append(unit)
        
    # Add common misspellings manually
    misspellings = {
        'הנדסה': ['הנדס', 'הנדסת', 'הנדסא', 'הנדצה', 'הנדסח', 'הנדה', 'הנדסס'],
        'מאב': ['מאפ', 'מאך', 'מעב', 'מאג', 'מאף', 'מאבב', 'מב'],
        'מהן': ['מהנ', 'מהל', 'מהכ', 'מהם', 'מחן', 'מהננ', 'מן'],
        'מעמ': ['מעכ', 'מאמ', 'מעל', 'מעם', 'מעג', 'מעממ', 'ממ'],
        'מתן': ['מתנ', 'מטן', 'מתכ', 'מתם', 'מתל', 'מתננ', 'מן'],
        'מטס': ['מטז', 'מטצ', 'מטש', 'מתס', 'מטק', 'מטסס', 'מס'],
        'תשתיות': ['תשתית', 'תשתיו', 'תשתיט', 'תשתיח', 'תשתיומ', 'תשת', 'תשתיותת']
    }
    
    for correct, wrong_list in misspellings.items():
        for wrong in wrong_list:
            training_data.append(wrong)
            labels.append(correct)
    
    # Create vectorizer and convert text to features
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))
    X = vectorizer.fit_transform(training_data)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, labels)
    
    return model, vectorizer

def correct_unit_spelling(unit_name, model, vectorizer):
    # If the unit name is already correct, return it
    correct_units = {'הנדסה', 'מאב', 'מהן', 'מעמ', 'מתן', 'מטס', 'תשתיות'}
    
    # Handle NaN values
    if pd.isna(unit_name):
        return "לא צוין"
    
    if unit_name in correct_units:
        return unit_name
    
    # Transform the input unit name
    X_test = vectorizer.transform([unit_name])
    
    # Predict the correct spelling
    corrected = model.predict(X_test)[0]
    
    return corrected

def clean_and_sort_data_with_model(input_file, allocations, unit_weights, model, vectorizer):
    df = pd.read_excel(input_file)
    cleaned_data = []
    assignments = []
    assigned_soldiers = set()

    # Check if all soldiers have pre-assignment
    if df['שיבוץ מקדים'].fillna('לא').eq('כן').all():
        raise ValueError("כל החיילים מסומנים כשיבוץ מקדים. אנא וודא שלפחות חלק מהחיילים אינם מסומנים.")

    # Track unit-level averages
    unit_averages = {unit: [] for unit in allocations.keys()}

    # First, handle pre-assigned soldiers
    pre_assigned_soldiers = df[df['שיבוץ מקדים'] == 'כן']
    for _, row in pre_assigned_soldiers.iterrows():
        soldier_name = row['שם החייל']
        city = row['עיר מגורים']
        
        # Handle potentially NaN value in 'יחידה מקדימה'
        if pd.isna(row['יחידה מקדימה']):
            continue  # Skip this soldier or handle it differently
            
        pre_assigned_unit = correct_unit_spelling(row['יחידה מקדימה'], model, vectorizer)
        
        # Find the soldier's rating for the pre-assigned unit
        unit_ratings = row['יחידות ודירוגים']
        if pd.isna(unit_ratings):
            continue  # Skip if ratings are missing
            
        soldier_unit_rating = None
        soldier_priority = None
        
        try:
            for priority, unit_rating in enumerate(unit_ratings.split('\n'), start=1):
                match = re.match(r"(.*)\((\d+)\)", unit_rating)
                if match:
                    unit_name = match.group(1).strip()
                    unit_name = correct_unit_spelling(unit_name, model, vectorizer)
                    unit_score = int(match.group(2))
                    if unit_name == pre_assigned_unit:
                        soldier_unit_rating = unit_score
                        soldier_priority = priority
                        break
        except AttributeError:
            # Handle case where unit_ratings cannot be split (e.g., not a string)
            continue
            
        soldier_unit_rating = soldier_unit_rating if soldier_unit_rating is not None else 1
        soldier_priority = soldier_priority if soldier_priority is not None else len(unit_ratings.split('\n')) if isinstance(unit_ratings, str) else 1
        
        if allocations.get(pre_assigned_unit, 0) > 0 and soldier_name not in assigned_soldiers:
            # Calculate unit and soldier weights
            unit_weight = unit_weights.get(pre_assigned_unit, 80)
            soldier_weight = 100 - unit_weight
            weight_unit = unit_weight / 100
            weight_soldier = soldier_weight / 100
            
            average_score = (weight_unit * soldier_unit_rating) + (weight_soldier * (8 - soldier_priority))
            
            assignment_entry = {
                'שם החייל': soldier_name,
                'עיר מגורים': city,
                'יחידה': pre_assigned_unit,
                'דירוג החייל': 8 - soldier_priority,
                'דירוג היחידה': soldier_unit_rating,
                'אחוז השפעה יחידה': unit_weight,
                'אחוז השפעה חייל': soldier_weight,
                'ממוצע': average_score
            }
            
            assignments.append(assignment_entry)
            unit_averages[pre_assigned_unit].append(soldier_unit_rating)
            
            allocations[pre_assigned_unit] -= 1
            assigned_soldiers.add(soldier_name)

    # Remove pre-assigned soldiers from the main dataframe
    df = df[~df['שם החייל'].isin(assigned_soldiers)]

    # Continue with existing assignment logic for remaining soldiers
    for _, row in df.iterrows():
        soldier_name = row['שם החייל']
        city = row['עיר מגורים']
        unit_ratings = row['יחידות ודירוגים']
        
        # Skip soldiers with missing rating data
        if pd.isna(unit_ratings):
            continue
            
        try:
            unit_rating_pairs = unit_ratings.split('\n')
        except AttributeError:
            # Handle case where unit_ratings cannot be split (not a string)
            continue

        for priority, unit_rating in enumerate(unit_rating_pairs, start=1):
            match = re.match(r"(.*)\((\d+)\)", unit_rating)
            if match:
                unit_name = match.group(1).strip()
                # Apply spell checking to the unit name
                unit_name = correct_unit_spelling(unit_name, model, vectorizer)
                unit_score = int(match.group(2))
                
                unit_weight = unit_weights.get(unit_name, 80)
                soldier_weight = 100 - unit_weight
                weight_unit = unit_weight / 100
                weight_soldier = soldier_weight / 100
                
                average_score = (weight_unit * unit_score) + (weight_soldier * (8 - priority))
                
                entry = {
                    'שם החייל': soldier_name,
                    'עיר מגורים': city,
                    'יחידה': unit_name,
                    'דירוג החייל': 8 - priority,
                    'דירוג היחידה': unit_score,
                    'אחוז השפעה יחידה': unit_weight,
                    'אחוז השפעה חייל': soldier_weight,
                    'ממוצע': average_score
                }
                
                cleaned_data.append(entry)

    # Sort and process assignments
    if cleaned_data:
        cleaned_df = pd.DataFrame(cleaned_data)
        cleaned_df = cleaned_df.sort_values(by='ממוצע', ascending=False)

        total_rounds = max(allocations.values()) if allocations else 0
        for round_number in range(total_rounds):
            used_units = set()
            for _, row in cleaned_df.iterrows():
                soldier_name = row['שם החייל']
                city = row['עיר מגורים']
                unit_name = row['יחידה']
                soldier_rating = row['דירוג החייל']
                unit_rating = row['דירוג היחידה']
                unit_weight = row['אחוז השפעה יחידה']
                soldier_weight = row['אחוז השפעה חייל']
                average_score = row['ממוצע']
                
                if (allocations.get(unit_name, 0) > 0 and 
                    unit_name not in used_units and 
                    soldier_name not in assigned_soldiers):
                    
                    assignment_entry = {
                        'שם החייל': soldier_name,
                        'עיר מגורים': city,
                        'יחידה': unit_name,
                        'דירוג החייל': soldier_rating,
                        'דירוג היחידה': unit_rating,
                        'אחוז השפעה יחידה': unit_weight,
                        'אחוז השפעה חייל': soldier_weight,
                        'ממוצע': average_score
                    }
                    
                    assignments.append(assignment_entry)
                    unit_averages[unit_name].append(unit_rating)
                    
                    allocations[unit_name] -= 1
                    used_units.add(unit_name)
                    assigned_soldiers.add(soldier_name)
            
            # Remove assigned soldiers from the dataframe
            cleaned_df = cleaned_df[~cleaned_df['שם החייל'].isin(assigned_soldiers)]
            
            if cleaned_df.empty:
                break

    # Add unit average to each assignment
    for assignment in assignments:
        unit_name = assignment['יחידה']
        assignment['ממוצע יחידה'] = (sum(unit_averages[unit_name]) / len(unit_averages[unit_name])) if unit_averages[unit_name] else 0

    # Create the output Excel file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    if assignments:
        assignments_df = pd.DataFrame(assignments)
        assignments_df.to_excel(temp_file.name, index=False)
    else:
        # Create an empty Excel file if no assignments were made
        pd.DataFrame().to_excel(temp_file.name, index=False)

    return assignments, temp_file.name

@app.route('/', methods=['GET', 'POST'])
def index():
    error_message = None
    success_message = None
    assignments = []
    temp_file_path = None

    if request.method == 'POST':
        try:
            # יצירת מודל תיקון שגיאות
            model, vectorizer = create_spelling_model()

            # משקלים לכל יחידה
            unit_weights = {
                'הנדסה': float(request.form['weight_unit_הנדסה']),
                'מאב': float(request.form['weight_unit_מאב']),
                'מהן': float(request.form['weight_unit_מהן']),
                'מעמ': float(request.form['weight_unit_מעמ']),
                'מתן': float(request.form['weight_unit_מתן']),
                'מטס': float(request.form['weight_unit_מטס']),
                'תשתיות': float(request.form['weight_unit_תשתיות'])
            }

            # הקצאות לכל יחידה
            allocations = {
                'הנדסה': int(request.form['הנדסה']),
                'מאב': int(request.form['מאב']),
                'מהן': int(request.form['מהן']),
                'מעמ': int(request.form['מעמ']),
                'מתן': int(request.form['מתן']),
                'מטס': int(request.form['מטס']),
                'תשתיות': int(request.form['תשתיות'])
            }

            # עיבוד קובץ
            file = request.files['file']
            if file:
                input_file = os.path.join('uploads', file.filename)
                os.makedirs('uploads', exist_ok=True)
                file.save(input_file)

                try:
                    assignments, temp_file_path = clean_and_sort_data_with_model(
                        input_file,
                        allocations,
                        unit_weights,
                        model,
                        vectorizer
                    )
                    # הצלחה - אם לא נזרקה שגיאה
                    success_message = "הקובץ עובד בהצלחה!"
                except ValueError as e:
                    error_message = str(e)
                except Exception as e:
                    if "np.nan is an invalid document" in str(e):
                        error_message = "שיבוץ מקדים - כולל את כל החיילים. אנא וודא שיש חיילים ללא שיבוץ מקדים."
                    else:
                        error_message = f"התרחשה שגיאה: {str(e)}"

        except Exception as e:
            error_message = f"התרחשה שגיאה כללית: {str(e)}"

    return render_template(
        'index.html',
        assignments=assignments,
        temp_file=temp_file_path,
        error_message=error_message,
        success_message=success_message
    )

@app.route('/download', methods=['GET'])
def download_file():
    temp_file = request.args.get('path')
    if temp_file and os.path.exists(temp_file):
        return send_file(temp_file, as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)