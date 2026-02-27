import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# 姿态数据
# quat = [0, -0.179422, 0.983772, 0]  # [qw, qx, qy, qz]
# position = [-0.250374, 0.0268222, 0.068]

position = [-0.00074868,  0.05364437,  0.07      ]
quat = [0., 0.98377216, 0.17942229, 0.        ]

# 转换为旋转矩阵
r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # [qx,qy,qz,qw]
rotation_matrix = r.as_matrix()
euler = r.as_euler('xyz', degrees=True)

print(f"欧拉角 (度): {euler}")
print(f"旋转矩阵:\n{rotation_matrix}")

# 创建3D图形 - 避免使用tight_layout
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 定义坐标轴长度
axis_length = 0.05

# 原点位置
origin = np.array(position)

# 绘制坐标轴
ax.quiver(origin[0], origin[1], origin[2], 
          rotation_matrix[0, 0], rotation_matrix[1, 0], rotation_matrix[2, 0], 
          length=axis_length, color='red', arrow_length_ratio=0.1, linewidth=3, label='X轴')

ax.quiver(origin[0], origin[1], origin[2], 
          rotation_matrix[0, 1], rotation_matrix[1, 1], rotation_matrix[2, 1], 
          length=axis_length, color='green', arrow_length_ratio=0.1, linewidth=3, label='Y轴')

ax.quiver(origin[0], origin[1], origin[2], 
          rotation_matrix[0, 2], rotation_matrix[1, 2], rotation_matrix[2, 2], 
          length=axis_length, color='blue', arrow_length_ratio=0.1, linewidth=3, label='Z轴')

# 绘制原点
ax.scatter(*origin, color='black', s=100, label='末端执行器位置')

# 绘制世界坐标系（参考）
world_axis_length = 0.03
ax.quiver(0, 0, 0, world_axis_length, 0, 0, color='pink', alpha=0.5, label='世界X')
ax.quiver(0, 0, 0, 0, world_axis_length, 0, color='lightgreen', alpha=0.5, label='世界Y')  
ax.quiver(0, 0, 0, 0, 0, world_axis_length, color='lightblue', alpha=0.5, label='世界Z')

# 设置图形属性
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.legend()
ax.set_title(f'末端执行器姿态可视化\n位置: {position}\n欧拉角: [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]°')

# 设置坐标轴范围
ax.set_xlim([-0.3, 0.1])
ax.set_ylim([-0.05, 0.1])
ax.set_zlim([0, 0.15])

# 手动调整布局而不使用tight_layout
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.show()
