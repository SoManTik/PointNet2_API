import numpy as np
import torch
import importlib
import sys
import os
import torch.nn.functional as F
import pandas as pd
from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
from werkzeug.utils import secure_filename
import json
import plotly
import pywavefront


def load_point_cloud_data(file_path):
    # Load the data from the .txt file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the data
    points = []
    for line in lines:
        # Split the line into individual values
        values = line.strip().split(',')

        x, y, z = map(float, values[:3])
 
        point = [x, y, z]
        points.append(point)

        points_arr = points

    points = np.array(points)
    # Reshape the points array to (24, 3, 1024)
    points = np.resize(points, (24, 3, 1024))

    return points_arr





UPLOAD_FOLDER = './PointNet2_API/uploads/'
ALLOWED_EXTENSIONS = {'txt'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



sys.path.append(os.path.join('./PointNet2_API/Pointnet_Pointnet2_pytorch/models'))




model_name = 'pointnet2_cls_ssg'
m = importlib.import_module(model_name)
model = m.get_model(40, normal_channel=False)


checkpoint = torch.load('./PointNet2_API/log/classification/pointnet2_cls_ssg/checkpoints/best_model.pth', map_location='cpu')
start_epoch = checkpoint['epoch']
model.load_state_dict(checkpoint['model_state_dict'])
classifier = model.eval()






app = Flask(__name__)

@app.route('/')
def upload():
 return render_template('upload.html')
 


@app.route('/upload_N_detect', methods=['POST'])
def upload_N_detect():

    if 'file' in request.files:
        file = request.files['file']
        # Get the list of files from webpage 
        files = request.files.getlist("file") 
  
        # Iterate for each file in the files List, and Save them 
        for file in files: 
            file.save(UPLOAD_FOLDER+file.filename) 

        # filename = secure_filename(file.filename)

        # # Here you should save the file
        # file_path = os.path.join(UPLOAD_FOLDER+str(filename))
        # # Remove leading whitespace before saving the file
        # content = file.read().decode('utf-8').lstrip()

        # with open(file_path, 'w') as f:
        #     f.write(content)

        file_arr = os.listdir('./PointNet2_API/uploads')
        file_obj_name = [string for string in file_arr if ".obj" in string]
        print("file_obj_name",file_obj_name)
        # file.save(UPLOAD_FOLDER+file.filename) 
        print(os.listdir('./PointNet2_API/uploads'))
        scene = pywavefront.Wavefront(UPLOAD_FOLDER+file_obj_name[0])
        vertices = scene.vertices
        df_obj= pd.DataFrame(vertices)
        file_path="./PointNet2_API/uploads/txt/uploaded.txt"
        df_obj.to_csv(file_path, sep=',', index=False,header=False)




        points_arr = load_point_cloud_data(file_path)

        df = pd.DataFrame(points_arr, columns=['x', 'y', 'z'])
        fig = px.scatter_3d(df, x='x', y='y', z='z')

        fig.update_traces(marker=dict(size=2))

        # Create graphJSON
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        


        # Initialize an empty array to store the data
        data = np.empty((24, 3, 1024))

        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                # Split the line into individual values using commas as delimiters
                values = line.strip().split(',')

                # Convert the values to floats and assign them to the data array
                data[i // 1024, :, i % 1024] = [float(value) for value in values[:3]]

        tensor = torch.from_numpy(data)
        output, _ = classifier(tensor.float())
        pred_choice = output.data.max(1)
        # probabilities = F.softmax(pred_choice.values, dim=0)



        data = {
        "probabilities":pred_choice[0],
        "class": pred_choice[1]
        }
        modelnet40_shape_names= ['airplane','bathtub','bed','bench', 
                                'bookshelf', 'bottle' ,'bowl' ,'car','chair','cone',
                                'cup','curtain','desk','door','dresser','flower_pot','glass_box',
                                'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                                'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                                'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        #load data into a DataFrame object:
        df = pd.DataFrame(data)
        probabilities_Each_class = pd.DataFrame(df.groupby('class').sum())


        sorted_probabilities_Each_class = probabilities_Each_class.sort_values(by='probabilities',ascending=False)
        print(sorted_probabilities_Each_class)
        predict_class = modelnet40_shape_names[sorted_probabilities_Each_class.head(1).index.values[0]]
        print(predict_class)

        # rm all files from ./PointNet2_API/uploads
        for filename in os.listdir('./PointNet2_API/uploads/'):
  
            if os.path.isfile(os.path.join('./PointNet2_API/uploads', filename)):
                
                os.remove(os.path.join('./PointNet2_API/uploads', filename))


        return render_template('result.html',graphJSON=graphJSON,prediction = predict_class)
    else:
        return 'No file uploaded'

# main driver function
if __name__ == '__main__':
 
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run()



