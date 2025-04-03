import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def calculate_transformation(point1, point2):
    """
    Calcule la transformation (translation et rotation) entre deux points
    
    :param point1: Premier point [x, y, z, qw, qx, qy, qz]
    :param point2: Deuxième point [x, y, z, qw, qx, qy, qz]
    :return: Dictionnaire avec translation, rotation et vecteur
    """
    # Extraction des positions et rotations
    pos1 = np.array(point1[:3])
    pos2 = np.array(point2[:3])
    
    # Calcul du vecteur entre les points
    vector = pos2 - pos1
    vector_length = np.linalg.norm(vector)
    
    # Conversion des quaternions en objets de rotation
    rot1 = R.from_quat(point1[3:])
    rot2 = R.from_quat(point2[3:])
    
    # Calcul de la rotation relative
    relative_rotation = rot2 * rot1.inv()
    
    return {
        'translation': vector,
        'translation_length': vector_length,
        'rotation': relative_rotation
    }

def visualize_point_pairs(point_pairs):
    """
    Visualise les points et calcule les transformations entre eux
    
    :param point_pairs: Liste de paires de points
    """
    # Création de la figure
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Pour stocker les informations de transformation
    transformations = []
    
    # Visualisation des points et calcul des transformations
    for pair in point_pairs:
        point1, point2 = pair
        
        # Extraction des positions
        x1, y1, z1 = point1[:3]
        x2, y2, z2 = point2[:3]
        
        # Conversion des quaternions en matrices de rotation
        rot1 = R.from_quat(point1[3:])
        rot2 = R.from_quat(point2[3:])
        
        # Tracé des points
        ax.scatter(x1, y1, z1, c='black', s=50)
        ax.scatter(x2, y2, z2, c='red', s=50)
        
        # Tracé du vecteur entre les points
        ax.quiver(x1, y1, z1, 
                  x2-x1, y2-y1, z2-z1, 
                  color='purple', 
                  label='Vecteur entre points')
        
        # Tracé de la direction principale pour chaque point
        main_direction_length = 0.5
        
        # Première orientation : matrice de rotation vers direction
        main_dir1 = rot1.apply([0, 0, 1])  # Axe Z par défaut
        ax.quiver(x1, y1, z1, 
                  main_dir1[0], main_dir1[1], main_dir1[2], 
                  color='orange', 
                  length=main_direction_length, 
                  normalize=True)
        
        # Deuxième orientation
        main_dir2 = rot2.apply([0, 0, 1])  # Axe Z par défaut
        ax.quiver(x2, y2, z2, 
                  main_dir2[0], main_dir2[1], main_dir2[2], 
                  color='orange', 
                  length=main_direction_length, 
                  normalize=True)
        
        # Calcul de la transformation
        transformation = calculate_transformation(point1, point2)
        transformations.append(transformation)
    
    # Configuration des axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Visualisation des Points et Transformations')
    
    # Ajustement automatique des limites
    all_points = [p for pair in point_pairs for p in pair]
    ax.set_xlim([min(p[0] for p in all_points)-0.1, max(p[0] for p in all_points)+0.1])
    ax.set_ylim([min(p[1] for p in all_points)-0.1, max(p[1] for p in all_points)+0.1])
    ax.set_zlim([min(p[2] for p in all_points)-0.1, max(p[2] for p in all_points)+0.1])
    
    plt.tight_layout()
    
    # Affichage des transformations
    print("Transformations entre les points :")
    for i, transfo in enumerate(transformations):
        print(f"\nTransformation {i+1}:")
        print("Vecteur de translation:", transfo['translation'])
        print("Longueur du vecteur:", transfo['translation_length'])
        
        # Conversion de la rotation en angle d'Euler
        euler_angles = transfo['rotation'].as_euler('xyz', degrees=True)
        print("Rotation (angles d'Euler en degrés):", euler_angles)
    
    plt.show()

# Exemple d'utilisation
exemple_point_pairs = [
    [
		[0.01,  0.35,   0.2,   -0.2363, 0.3032, 0.8525, 0.3541],
		[0.038, 0.2374, 0.295, -0.236,  0.303,  0.852,  0.354]
    ],
    [
        [0.01,  0.35,  0.2,   -0.4271, -0.1594, 0.8485, 0.2686],
   		[0.131, 0.301, 0.273, -0.427,  -0.159,  0.849,  0.268]
    ]
]

visualize_point_pairs(exemple_point_pairs)