from flask import Flask, request, render_template, redirect, url_for
import os
import json
from detect import load_image_into_numpy_array, detect_objects, draw_boxes, process_video, model

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
DETECTIONS_FILE = 'detected_objects.json'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

detected_objects = {}

def load_detected_objects():
    global detected_objects
    if os.path.exists(DETECTIONS_FILE):
        with open(DETECTIONS_FILE, 'r') as f:
            detected_objects = json.load(f)
            print("Detected objects loaded from file")
    else:
        detected_objects = {}

def save_detected_objects():
    with open(DETECTIONS_FILE, 'w') as f:
        json.dump(detected_objects, f)
        print("Detected objects saved to file")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            print("No file part in request")
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            print("No selected file")
            return redirect(request.url)
        
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            print(f"File saved to {filepath}")
            
            if file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
                output_path, detected_classes = process_video(filepath, filepath.replace('.mp4', '_processed.mp4').replace('.avi', '_processed.avi').replace('.mov', '_processed.mov'))
                print(f"Processed video saved to {output_path}")
                detected_objects[file.filename] = {
                    'type': 'video',
                    'path': os.path.relpath(output_path, 'static').replace("\\", "/"),
                    'objects': detected_classes  # Store detected objects
                }
            else:
                image_np = load_image_into_numpy_array(filepath)
                detections = detect_objects(image_np)
                boxed_image_path = draw_boxes(filepath, detections)
                print(f"Boxes drawn on {boxed_image_path}")
                
                objects = [model.names[int(det[5])] for det in detections if det[4] > 0.5]
                detected_objects[file.filename] = {
                    'type': 'image',
                    'path': os.path.relpath(boxed_image_path, 'static').replace("\\", "/"),
                    'objects': objects
                }
                print(f"Detected objects for {file.filename}: {objects}")
            
            save_detected_objects()
            print("Detections saved")
            print(detected_objects)
            return redirect(url_for('home'))
        
        print("Redirecting back to home")
        return redirect(request.url)
    except Exception as e:
        print(f"Error during upload: {e}")
        return redirect(request.url)

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    results = []
    for filename, metadata in detected_objects.items():
        print(f"Checking file: {filename}, metadata: {metadata}")
        if any(query.lower() in obj.lower() for obj in metadata['objects']):
            results.append(metadata)
            print(f"Match found: {metadata}")
    return render_template('index.html', results=results, query=query)

if __name__ == '__main__':
    load_detected_objects()
    app.run(debug=True)