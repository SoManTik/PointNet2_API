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
import numpy as np
import open3d as o3d
import plotly.graph_objects as go


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


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





UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = {'txt'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



sys.path.append(os.path.join('./Pointnet_Pointnet2_pytorch/models'))




model_name = 'pointnet2_cls_ssg'
m = importlib.import_module(model_name)
model = m.get_model(40, normal_channel=False)


checkpoint = torch.load('./log/classification/pointnet2_cls_ssg/checkpoints/best_model.pth', map_location='cpu')
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
        print(files)
        try:
            # Iterate for each file in the files List, and Save them 
            for file in files: 
                file.save(UPLOAD_FOLDER+file.filename) 
        except Exception:
            return "<h1>No files uploaded</h1>"

        file_arr = os.listdir('./uploads')
        file_obj_name = [string for string in file_arr if ".obj" in string]


        scene = pywavefront.Wavefront(UPLOAD_FOLDER+file_obj_name[0])
        vertices = scene.vertices
        df_obj= pd.DataFrame(vertices)
        file_path="./uploads/txt/uploaded.txt"
        point_cloud = df_obj.to_numpy()
        point_cloud = pc_normalize(point_cloud)
        point_cloud = farthest_point_sample(point_cloud,len(point_cloud))

        # if len(point_cloud)>10000:
        #       point_cloud = point_cloud[np.random.choice(len(point_cloud), 10000, replace=False)]
        df_obj = pd.DataFrame(point_cloud)

        df_obj.to_csv(file_path, sep=',', index=False,header=False)




        points_arr = load_point_cloud_data(file_path)

        df = pd.DataFrame(points_arr, columns=['x', 'y', 'z'])
        fig = px.scatter_3d(df, x='x', y='y', z='z')

        fig.update_traces(marker=dict(size=2))

        # Create graphJSON point cloud
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        mesh = o3d.io.read_triangle_mesh(UPLOAD_FOLDER+file_obj_name[0])


        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        colors = None
        if mesh.is_empty(): exit()
        if not mesh.has_vertex_normals(): mesh.compute_vertex_normals()
        if not mesh.has_triangle_normals(): mesh.compute_triangle_normals()
        if mesh.has_triangle_normals():
            colors = (0.5, 0.5, 0.5) + np.asarray(mesh.triangle_normals) * 0.5
            colors = tuple(map(tuple, colors))
        else:
            colors = (1.0, 0.0, 0.0)

        fig = go.Figure(
            data=[
            
                go.Mesh3d(
                    x=vertices[:,0],
                    y=vertices[:,1],
                    z=vertices[:,2],
                    i=triangles[:,0],
                    j=triangles[:,1],
                    k=triangles[:,2],
                    facecolor=colors,
                    opacity=0.50)
            ],
            layout=dict(
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False)
                )
            )
        )
        graphJSON_with_texture = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


        


        # Initialize an empty array to store the data
        data = np.empty((24, 3, 1024))

        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                # Split the line into individual values using commas as delimiters
                values = line.strip().split(',')

                # Convert the values to floats and assign them to the data array
                try:
                    data[i // 1024, :, i % 1024] = [float(value) for value in values[:3]]
                except Exception:
                    pass


        tensor = torch.from_numpy(data)
        output, _ = classifier(tensor.float())
        pred_choice = output.data.max(1)
        probabilities = F.softmax(pred_choice.values, dim=0)



        data = {
        "score":pred_choice[0],
        "class": pred_choice[1]
        }
        modelnet40_shape_names= ['airplane','bathtub','bed','bench', 'bookshelf', 
                                 'bottle' ,'bowl' ,'car','chair','cone',
                                'cup','curtain','desk','door','dresser',
                                'flower_pot','glass_box','guitar','keyboard','lamp',
                                'laptop','mantel','monitor','night_stand',
                                'person','piano','plant','radio','range_hood',
                                'sink','sofa','stairs','stool','table',
                                'tent','toilet','tv_stand','vase',
                                'wardrobe','xbox']

        #load data into a DataFrame object:
        df = pd.DataFrame(data)
        probabilities_Each_class = pd.DataFrame(df.groupby('class').sum())


        sorted_probabilities_Each_class = probabilities_Each_class.sort_values(by='score',ascending=False)
        print(sorted_probabilities_Each_class)
        predict_class = modelnet40_shape_names[sorted_probabilities_Each_class.head(1).index.values[0]]
 
        # rm all files from ./uploads
        for filename in os.listdir('./uploads/'):
            print(filename)
            if os.path.isfile(os.path.join('./uploads', filename)):
                
                os.remove(os.path.join('./uploads', filename))


        return render_template('result.html',graphJSON=graphJSON, graphJSON_with_texture= graphJSON_with_texture, prediction = predict_class)
    else:
        return 'No file uploaded'

# main driver function
if __name__ == '__main__':
 
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(debug=True,host="0.0.0.0",port=3000)

