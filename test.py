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

# Selecionar os pontos no plano
inlier_cloud = pcd.select_by_index(inliers)

# Selecionar os pontos fora do plano
outlier_cloud = pcd.select_by_index(inliers, invert=True)

# Selecionar os pontos que estão acima do plano
above_plane_indices = np.where(a * np.array(outlier_cloud.points)[:, 0] + b * np.array(outlier_cloud.points)[:, 1] + c * np.array(outlier_cloud.points)[:, 2] + d > 0)[0]
above_plane_cloud = outlier_cloud.select_by_index(above_plane_indices)

# Projetar os pontos acima do plano para o plano
projected_points = []
for point in np.asarray(above_plane_cloud.points):
    t = -(a*point[0] + b*point[1] + c*point[2] + d) / (a*a + b*b + c*c)
    projected_point = point + np.array([a, b, c]) * t
    projected_points.append(projected_point)

# Criar uma nova nuvem de pontos para os pontos projetados
above_plane_cloud_projected = o3d.geometry.PointCloud()
above_plane_cloud_projected.points = o3d.utility.Vector3dVector(projected_points)

# Visualizar a nuvem de pontos
o3d.visualization.draw_geometries([above_plane_cloud_projected])

# Calcular a distância de cada ponto no inlier_cloud aos pontos acima do plano
distances = inlier_cloud.compute_point_cloud_distance(above_plane_cloud_projected)

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
o3d.visualization.draw_geometries([red_cloud, blue_cloud, above_plane_cloud_projected, above_plane_cloud])