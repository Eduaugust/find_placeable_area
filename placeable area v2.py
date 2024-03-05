import time
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
import pymesh
import trimesh

def point_cloud_to_mesh(pcd):
    # Estimar normais
    pcd.estimate_normals()

    # Calcular malha usando o algoritmo Ball-Pivoting
    radii = [0.005, 0.01, 0.02, 0.04]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
               pcd,
               o3d.utility.DoubleVector(radii))

    return mesh

def translate_points_from_center(pcd, plane_model, distance=0.1):
    # Obter o vetor normal ao plano
    normal_vector = np.array(plane_model[:3])

    # Calcular o vetor de translação
    translation_vector = normal_vector * distance

    # Obter os pontos da nuvem de pontos
    points = np.asarray(pcd.points)

    # Calcular o centro dos pontos
    center = np.mean(points, axis=0)

    # Transladar os pontos a partir do centro
    translated_points = points + (points - center) + translation_vector

    # Criar uma nova nuvem de pontos para os pontos traduzidos
    translated_pcd = o3d.geometry.PointCloud()
    translated_pcd.points = o3d.utility.Vector3dVector(translated_points)

    return translated_pcd

def translate_points(pcd, plane_model, distance=0.01):
    # Obter o vetor normal ao plano
    normal_vector = np.array(plane_model[:3])

    # Calcular o vetor de translação
    translation_vector = normal_vector * distance

    # Obter os pontos da nuvem de pontos
    points = np.asarray(pcd.points)

    # Transladar os pontos
    translated_points = points + translation_vector

    # Criar uma nova nuvem de pontos para os pontos traduzidos
    translated_pcd = o3d.geometry.PointCloud()
    translated_pcd.points = o3d.utility.Vector3dVector(translated_points)

    return translated_pcd

def get_edges(pcd):
    translated_pcd = translate_points(pcd, plane_model, 0.0001)

    combined_pcd = pcd + translated_pcd
    edges = hull(combined_pcd)

    return edges

def grow_region(projected_cloud_obj, obj_pcd, distance_threshold=0.01):
    # Obter os pontos do objeto projetado
    projected_points = np.asarray(projected_cloud_obj.points)

    # Inicializar a região com o ponto inicial
    region_indices = [0]
    region_points = [projected_points[0]]

    # Crescer a região até que tenha o mesmo número de pontos que o objeto
    while len(region_indices) < len(obj_pcd.points):
        # Calcular a distância de cada ponto aos pontos na região
        distances = np.linalg.norm(projected_points - region_points[-1], axis=1)
        
        # Adicionar o ponto mais próximo que ainda não está na região
        min_distance_index = np.argmin(distances)
        if min_distance_index not in region_indices and distances[min_distance_index] < distance_threshold:
            region_indices.append(min_distance_index)
            region_points.append(projected_points[min_distance_index])

    # Criar uma nova nuvem de pontos para a região
    region_cloud = o3d.geometry.PointCloud()
    region_cloud.points = o3d.utility.Vector3dVector(region_points)

    return region_cloud

def get_projected_cloud(plane_model, cloud):
    [a, b, c, d] = plane_model
    projected_points = []
    for point in np.array(cloud.points):
        t = -(a*point[0] + b*point[1] + c*point[2] + d) / (a*a + b*b + c*c)
        projected_point = point + np.array([a, b, c]) * t
        projected_points.append(projected_point)
    projected_cloud = o3d.geometry.PointCloud()
    projected_cloud.points = o3d.utility.Vector3dVector(projected_points)
    return projected_cloud

def extract_color(pc):
    # Encontrar cores únicas e suas contagens
    unique_colors, counts = np.unique(pc, axis=0, return_counts=True)

    # Imprimir o número de cores únicas
    print(f"Existem {len(unique_colors)} cores únicas.")

    # Imprimir as cores únicas
    for i, color in enumerate(unique_colors):
        print(f"Cor {i+1}: {color * 255} {color}")

def clustering():
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd])
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # view_color_pallet()
    extract_color(colors)

def extract_obj():
    obj_color = np.array([0.83921569, 0.15294118, 0.15686275])

    tolerance = 0.1
    colors = np.asarray(pcd.colors)
    indices = np.where(np.linalg.norm(colors - obj_color, axis=1) < tolerance)[0]

    obj_pcd = pcd.select_by_index(indices)

    # o3d.visualization.draw_geometries([obj_pcd])
    return obj_pcd

def hull(obj_pcd):
    hull, _ = obj_pcd.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
    # o3d.visualization.draw_geometries([hull_ls])

    return hull_ls

def view_color_pallet():
    # Gerar as cores
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max_label + 1))

    # Criar uma figura e um eixo
    fig, ax = plt.subplots(1, 1, figsize=(5, 2),
                            dpi=80, facecolor='w', edgecolor='k')

    # Para cada cor, desenhar um retângulo preenchido com a cor e adicionar um texto com o rótulo correspondente
    for sp in range(max_label + 1):
        rect = plt.Rectangle((sp, 0), 1, 1, facecolor=colors[sp])
        ax.add_patch(rect)
        ax.annotate(f'Cluster {sp}', (sp + 0.5, 0.5), color='black', weight='bold', 
                    fontsize=6, ha='center', va='center')

    # Remover os eixos
    ax.set_xlim([0, max_label + 1])
    ax.set_ylim([0, 1])
    ax.axis('off')

    # Mostrar a legenda de cores
    plt.show()

    # Visualizar a nuvem de pontos
    o3d.visualization.draw_geometries([pcd])

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

above_plane_cloud_projected = get_projected_cloud(plane_model, above_plane_cloud)
# o3d.visualization.draw_geometries([above_plane_cloud_projected])

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
# o3d.visualization.draw_geometries([red_cloud, blue_cloud, above_plane_cloud_projected, above_plane_cloud])

pcd = above_plane_cloud

clustering()

obj_pcd = extract_obj()

hull(obj_pcd)

projected_cloud_obj = get_projected_cloud(plane_model, obj_pcd)

translated_pcd = translate_points_from_center(obj_pcd, plane_model)

# Converter nuvens de pontos em malhas
mesh1 = point_cloud_to_mesh(obj_pcd)
mesh2 = point_cloud_to_mesh(translated_pcd)

# Converter malhas do Open3D para malhas trimesh
trimesh_mesh1 = trimesh.Trimesh(np.asarray(mesh1.vertices), np.asarray(mesh1.triangles))
trimesh_mesh2 = trimesh.Trimesh(np.asarray(mesh2.vertices), np.asarray(mesh2.triangles))

# Calcular a diferença booleana
difference_mesh = trimesh_mesh1.difference(trimesh_mesh2)

# Converter a malha de diferença de volta para uma malha Open3D para visualização
difference_mesh_o3d = o3d.geometry.TriangleMesh()
difference_mesh_o3d.vertices = o3d.utility.Vector3dVector(difference_mesh.vertices)
difference_mesh_o3d.triangles = o3d.utility.Vector3iVector(difference_mesh.faces)

# Visualizar a malha de diferença
o3d.visualization.draw_geometries([difference_mesh_o3d])