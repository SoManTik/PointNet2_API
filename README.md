# PointNet2_API
web_application of PointNet2 is a Flask web application.


You can access below website to generate point obj, but there are some rules you must follow:

1. The generation of 3D point cloud should be retopology from https://lumalabs.ai.

2. Please to select low option for generation, as the model was trained on ModelNet40, each obj contains approximately 10000 points. Therefore, middle and high options will generate high quality and size which is NOT suitable for the model.

3. Please clone the repository first.
```bash
cd PointNet2_API/
```

4. 
```bash
git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch
```

5. run main.py



Folder structure:

```bash
.
├── PointNet2_API
│   ├── Pointnet_Pointnet2_pytorch
│   ├── README.md
│   ├── Testing.py
│   ├── data
│   ├── log
│   ├── templates
│   └── uploads
```




Reference:
https://github.com/yanx27/Pointnet_Pointnet2_pytorch
