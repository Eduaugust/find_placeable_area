import time
import numpy as np
import open3d as o3d

# Ler a nuvem de pontos
pcd = o3d.io.read_point_cloud("points.ply")

# measure time
start = time.time()

# Segmentar o plano
plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# Selecionar e pintar os pontos no plano
inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])

# Selecionar os pontos fora do plano
outlier_cloud = pcd.select_by_index(inliers, invert=True)

# Selecionar os pontos que estão acima do plano
above_plane_indices = np.where(a * np.array(outlier_cloud.points)[:, 0] + b * np.array(outlier_cloud.points)[:, 1] + c * np.array(outlier_cloud.points)[:, 2] + d > 0)[0]
above_plane_cloud = outlier_cloud.select_by_index(above_plane_indices)

# Calcular a distância de cada ponto no inlier_cloud aos pontos acima do plano
distances = inlier_cloud.compute_point_cloud_distance(above_plane_cloud)

# Selecionar os pontos no inlier_cloud que estão a uma distância menor que 10 centímetros de qualquer ponto acima do plano
indices = np.where(np.array(distances) < 0.1)[0]

# Pintar esses pontos de vermelho
red_cloud = inlier_cloud.select_by_index(indices)
red_cloud.paint_uniform_color([1.0, 0, 0])

# Selecionar os pontos restantes no inlier_cloud e pintar de azul
blue_cloud = inlier_cloud.select_by_index(indices, invert=True)
blue_cloud.paint_uniform_color([0, 0, 1.0])

# measure time
end = time.time()
print(f"Runtime of the program is {end - start}")

print(f"Total points: {len(pcd.points)}")

# Visualizar a nuvem de pontos
o3d.visualization.draw_geometries([red_cloud, blue_cloud, above_plane_cloud])