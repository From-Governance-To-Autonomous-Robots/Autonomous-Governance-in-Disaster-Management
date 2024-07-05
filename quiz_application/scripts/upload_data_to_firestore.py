import firebase_admin
from firebase_admin import credentials, firestore, storage
import os

# Initialize Firebase Admin SDK
cred = credentials.Certificate('disaster-master.json')  # Replace with your Firebase Admin SDK key
firebase_admin.initialize_app(cred, {
    'storageBucket': 'disaster-master-59d6f.appspot.com'  # Replace with your storage bucket URL
})

db = firestore.client()
bucket = storage.bucket()

# Function to upload image to Firebase Storage
def upload_image(image_path):
    blob = bucket.blob(os.path.basename(image_path))
    blob.upload_from_filename(image_path)
    blob.make_public()
    return blob.public_url

# Function to upload question to Firestore
def upload_question(question_id, image_path, text, task, correct_answer, question_options, question_format):
    image_url = upload_image(image_path)
    question_data = {
        'image': image_url,
        'text': text,
        'task': task,
        'correct_answer': correct_answer,
        'question_options': question_options,
        'question_format': question_format
    }
    db.collection('questions').document(question_id).set(question_data)

# Example usage
upload_question(
    question_id='q1',
    image_path='test.png',
    text='What is shown in the image?',
    task='identify_object',
    correct_answer='apple',
    question_options=['apple', 'banana', 'cherry', 'date'],
    question_format='multiple_choice'
)
