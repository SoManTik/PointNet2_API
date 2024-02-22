# PointNet2_API
web_application of PointNet2


You may access below website to generate point obj, but there are some rules to must follow it:
    1.The generation of 3D point cloud shd be retopology.
    2.select low ooption for the generation since the model was trained on ModelNet40, each obj approximately contain 10000 points. Therefore, middle and high option will create high quality and size that is NOT suitable for the model.
    3.cd PointNet2_API/
    4.git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch
    


Folder structure:

```bash
.
├── PointNet2_API
│   ├── Pointnet_Pointnet2_pytorch
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── data_utils
│   │   │   ├── ModelNetDataLoader.py
│   │   │   ├── S3DISDataLoader.py
│   │   │   ├── ShapeNetDataLoader.py
│   │   │   ├── collect_indoor3d_data.py
│   │   │   ├── indoor3d_util.py
│   │   │   └── meta
│   │   │       ├── anno_paths.txt
│   │   │       └── class_names.txt
│   │   ├── log
│   │   │   ├── classification
│   │   │   │   ├── pointnet2_msg_normals
│   │   │   │   │   ├── checkpoints
│   │   │   │   │   │   └── best_model.pth
│   │   │   │   │   ├── logs
│   │   │   │   │   │   └── pointnet2_cls_msg.txt
│   │   │   │   │   ├── pointnet2_cls_msg.py
│   │   │   │   │   └── pointnet2_utils.py
│   │   │   │   └── pointnet2_ssg_wo_normals
│   │   │   │       ├── checkpoints
│   │   │   │       │   └── best_model.pth
│   │   │   │       ├── logs
│   │   │   │       │   └── pointnet2_cls_ssg.txt
│   │   │   │       ├── pointnet2_cls_ssg.py
│   │   │   │       └── pointnet2_utils.py
│   │   │   ├── part_seg
│   │   │   │   └── pointnet2_part_seg_msg
│   │   │   │       ├── checkpoints
│   │   │   │       │   └── best_model.pth
│   │   │   │       ├── logs
│   │   │   │       │   └── pointnet2_part_seg_msg.txt
│   │   │   │       ├── pointnet2_part_seg_msg.py
│   │   │   │       └── pointnet2_utils.py
│   │   │   └── sem_seg
│   │   │       ├── pointnet2_sem_seg
│   │   │       │   ├── checkpoints
│   │   │       │   │   └── best_model.pth
│   │   │       │   ├── eval.txt
│   │   │       │   ├── logs
│   │   │       │   │   └── pointnet2_sem_seg.txt
│   │   │       │   ├── pointnet2_sem_seg.py
│   │   │       │   └── pointnet2_utils.py
│   │   │       └── pointnet_sem_seg
│   │   │           ├── checkpoints
│   │   │           │   └── best_model.pth
│   │   │           ├── eval.txt
│   │   │           ├── logs
│   │   │           │   └── pointnet_sem_seg.txt
│   │   │           ├── pointnet2_utils.py
│   │   │           └── pointnet_sem_seg.py
│   │   ├── models
│   │   │   ├── __pycache__
│   │   │   │   ├── pointnet2_cls_ssg.cpython-310.pyc
│   │   │   │   └── pointnet2_utils.cpython-310.pyc
│   │   │   ├── pointnet2_cls_msg.py
│   │   │   ├── pointnet2_cls_ssg.py
│   │   │   ├── pointnet2_part_seg_msg.py
│   │   │   ├── pointnet2_part_seg_ssg.py
│   │   │   ├── pointnet2_sem_seg.py
│   │   │   ├── pointnet2_sem_seg_msg.py
│   │   │   ├── pointnet2_utils.py
│   │   │   ├── pointnet_cls.py
│   │   │   ├── pointnet_part_seg.py
│   │   │   ├── pointnet_sem_seg.py
│   │   │   └── pointnet_utils.py
│   │   ├── provider.py
│   │   ├── test_classification.py
│   │   ├── test_partseg.py
│   │   ├── test_semseg.py
│   │   ├── train_classification.py
│   │   ├── train_partseg.py
│   │   ├── train_semseg.py
│   │   └── visualizer
│   │       ├── build.sh
│   │       ├── eulerangles.py
│   │       ├── pc_utils.py
│   │       ├── pic.png
│   │       ├── pic2.png
│   │       ├── plyfile.py
│   │       ├── render_balls_so.cpp
│   │       └── show3d_balls.py
│   ├── README.md
│   ├── Testing.py
│   ├── data
│   │   └── modelNet40_testingData
│   │       ├── airplane_0640.txt
│   │       ├── bathtub_0156.txt
│   │       ├── bed_0517.txt
│   │       ├── door_0001.txt
│   │       ├── guitar_0067.txt
│   │       └── keyboard_0001.txt
│   ├── log
│   │   └── classification
│   │       └── pointnet2_cls_ssg
│   │           ├── checkpoints
│   │           │   └── best_model.pth
│   │           ├── logs
│   │           │   └── pointnet2_cls_ssg.txt
│   │           ├── pointnet2_cls_ssg.py
│   │           ├── pointnet2_utils.py
│   │           └── train_classification.py
│   ├── templates
│   │   ├── result.html
│   │   └── upload.html
│   └── uploads
├── obj.txt
```




Reference:
https://github.com/yanx27/Pointnet_Pointnet2_pytorch
