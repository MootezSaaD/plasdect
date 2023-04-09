import os
import shutil
from subprocess import check_output
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compute', methods=['POST'])
def compute_code_smell():
    # Save uploaded file to temporary folder
    console.log(request.files)
    print(request.files)
    file = request.files['java-file']
    temp_dir = 'temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, file.filename)
    file.save(file_path)
    
    # Run jar file on temporary folder
    jar_file = 'CodeSplitJava.jar'
    output_dir = 'output'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    method = 'class'
    command = f'java -jar {jar_file} -i "{file_path}" -o "{output_dir}" -m "{method}"'
    check_output(command, shell=True)
    
    # Send output to GNN model for inference
    # ...

    # Return success message to user
    return 'Code smell computed successfully'

if __name__ == '__main__':
    app.run(debug=True)
