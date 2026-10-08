import cv2
import os
import numpy as np
import transforms3d as t3d
import open3d as o3d
from scipy.spatial import ConvexHull
import scipy.linalg
from shapely.geometry import Polygon
from shapely.validation import make_valid
import matplotlib.pyplot as plt
from model.runner.base_runner import BaseRunner

class BaseStackMaskdepthRunner(BaseRunner):
    def __init__(self,
                 intrinsics,
                 raw_endpose=None,
                 calib_endpose=None,
                 ):
        super().__init__()
        self.intrinsics = intrinsics
        self.Transfer = self.cal_transfer(raw_endpose,calib_endpose)

    ### transfer
    # quat:[rx,ry,rz,rw]
    def quat2R(self,quat):
        quat_trans = [quat[-1],quat[0],quat[1],quat[2]] # trans to [rw,rx,ry,rz]
        rot_matrix = t3d.quaternions.quat2mat(quat_trans)
        return rot_matrix

    # 末端执行点的位姿表示base坐标系到末端执行器坐标系的转移矩阵。
    # 反过来，这个转移矩阵作为点的坐标转移矩阵就应该将是将末端执行器坐标转移到base坐标系的坐标
    # pose: [x,y,z,rx,ry,rz,rw]
    def cal_transfer(self,raw_endpose,calib_endpose):
        if raw_endpose is None or calib_endpose is None or raw_endpose==calib_endpose:
            return np.eye(4, dtype=np.float32)
        
        R_s = self.quat2R(raw_endpose[-4:])
        t_s = raw_endpose[:3]
        T_s = t3d.affines.compose(t_s, R_s, [1,1,1]) # compose(平移, 旋转, 缩放)，缩放设为1

        R_e = self.quat2R(calib_endpose[-4:])
        t_e = calib_endpose[:3]
        T_e = t3d.affines.compose(t_e, R_e, [1,1,1])

        T_e_inv = scipy.linalg.inv(T_e)
        Transfer = T_e_inv @ T_s

        return Transfer

    ### maskdepth2polygon
    def mapping(self,maskdepth,intrinsics=None):
        if intrinsics is None:
            intrinsics = self.intrinsics
        height, width = maskdepth.shape
    
        # 获取相机内参
        fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
        
        # 生成像素坐标网格
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        
        # 根据深度图计算对应的3D坐标
        z_coords = maskdepth / 1000.0  # 假设深度单位是毫米，转换为米
        x_coords = (x_coords - cx) * z_coords / fx
        y_coords = (y_coords - cy) * z_coords / fy
        
        # 将 3D 坐标转化为点 (x, y, z)
        points = np.stack((x_coords, y_coords, z_coords), axis=-1)
        
        # 扁平化点云
        pointcloud = points.reshape((-1, 3))
        return pointcloud

    # project
    def project(self,pointcloud,Transfer=None):
        if Transfer is None:
            return pointcloud
        pointcloud = np.hstack([pointcloud, np.ones((pointcloud.shape[0], 1))]) # append dim
        pointcloud_transfer = np.dot(pointcloud, Transfer.T)
        return pointcloud_transfer[:,:3]

    def hull_polygon(self,pointcloud2d):
        if len(pointcloud2d)<3:
            raise ValueError("Nums of pointcloud <3")
        hull = ConvexHull(pointcloud2d)
        hull_points = pointcloud2d[hull.vertices]
        
        # if not np.allclose(hull_points[0], hull_points[-1]):
        #     hull_points = np.vstack([hull_points, hull_points[0]])  # 闭合多边形
        
        polygon = make_valid(Polygon(hull_points))
        return polygon

    def maskdepth2polygon(self,maskdepth,Transfer=None,out_pointcloud=False):
        if Transfer is None:
            Transfer = self.Transfer
        pointcloud = self.mapping(maskdepth)
        pointcloud_transfer = self.project(pointcloud,Transfer)
        polygon = self.hull_polygon(pointcloud_transfer[:,:2])
        if out_pointcloud:  
            return polygon,pointcloud_transfer
        return polygon

    ### polygon_step

    # def cal_pointcloud2d_area(self,polygon):
    #     ## 也可用hullpoints鞋带公式实现
    #     # x = hull_points[:, 0]
    #     # y = hull_points[:, 1]
    #     # area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))

    #     area = polygon.area()
    #     return area

    def cal_center(self,polygon):
        center = polygon.representative_point()
        return [center.x,center.y]

    def cal_overlap_area(self,polygon1,polygon2):
        if polygon1.is_empty or polygon2.is_empty:
            overlap_area = 0.0
        else:
            intersection_polygon = polygon1.intersection(polygon2)
            overlap_area = intersection_polygon.area
        return overlap_area

    def cal_union_area(self,polygon1,polygon2):
        if polygon1.is_empty or polygon2.is_empty:
            union_area = 0.0
        else:
            union_poly = polygon1.union(polygon2)
            union_area = union_poly.area
        return union_area

    def cal_roi(self,polygon1,polygon2):
        overlap_area = self.cal_overlap_area(polygon1,polygon2)
        union_area = self.cal_union_area(polygon1,polygon2)
        roi = overlap_area/union_area if union_area != 0.0 else 0.0
        return roi

    def cal_d(self,polygon1,polygon2):
        center1 = self.cal_center(polygon1)
        center2 = self.cal_center(polygon2)

        d = np.sqrt((center1[0]-center2[0])**2+(center1[1]-center2[1])**2)
        return d
    
    def cal_score(self,polygon1,polygon2,weight=1.0):
        roi = self.cal_roi(polygon1,polygon2)
        d = self.cal_d(polygon1,polygon2)
        return roi*weight+d*(1.0-weight)

    def cal_step(self,polygon0,polygon1,lr=0.02,weight=1.0,thread=0.06):
        score = self.cal_score(polygon0,polygon1,weight)
        if score > thread:
            return [True,score]
        else:
            center0 = self.cal_center(polygon0)
            center1 = self.cal_center(polygon1)
            l2 = np.sqrt((center0[0]-center1[0])**2+(center0[1]-center1[1])**2)
            x = (center0[0]-center1[0])/l2*lr
            y = (center0[1]-center1[1])/l2*lr
            return [False,score,x,y]
    
    ### show
    def show(self,inputs,type):
        if type == 'pointcloud':
            points = np.vstack(inputs)
            pointcloud = o3d.geometry.PointCloud()
            pointcloud.points = o3d.utility.Vector3dVector(points)
            o3d.visualization.draw_geometries([pointcloud])
        elif type == 'polygon':
            fig, ax = plt.subplots(figsize=(8, 8))
            for idx, polygon in enumerate(inputs):
                x, y = polygon.exterior.xy
                
                # 随机颜色
                outline_color = (np.random.random(), np.random.random(), np.random.random())
                fill_color = (*outline_color, 0.5)
                
                # 绘制当前多边形的轮廓（添加序号标签，方便区分）
                ax.plot(x, y, color=outline_color, linewidth=2, label=f'Polygon {idx+1}')
                # 填充当前多边形的内部
                ax.fill(x, y, color=fill_color)
            # 4. 关键设置：保证坐标轴等比例（避免图形变形）
            ax.set_aspect('equal')
            # 可选美化：网格、图例（显示所有多边形）、标题
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='best')  # 自动选择最优图例位置
            ax.set_title('Polygons')
            # 5. 自动调整坐标轴范围（确保所有多边形都完整显示）
            ax.autoscale()
            plt.show()

    def setup(self):
        raise NotImplementedError()
    
    def step(self):
        raise NotImplementedError()