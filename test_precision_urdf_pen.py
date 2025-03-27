import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def calculate_z0_point_from_quaternion(point, quaternion):
    """
    Calcule le point sur Z = 0 qui, connecté au point original, 
    respecte l'orientation du quaternion.
    
    Parameters:
    point (np.array): Point 3D original
    quaternion (np.array): Quaternion définissant l'orientation
    
    Returns:
    np.array: Point projeté sur Z = 0
    """
    # Convertir le quaternion en matrice de rotation
    rotation = R.from_quat(quaternion)
    
    # Vecteur de direction initial (axe Z)
    initial_vector = np.array([0, 0, 1])
    
    # Appliquer la rotation au vecteur initial
    rotated_vector = rotation.apply(initial_vector)
    
    # Paramètres de l'équation de la ligne
    # Ligne passant par 'point' dans la direction de 'rotated_vector'
    # Équation : point_z0 = point + t * rotated_vector
    # On cherche t tel que point_z0[2] = 0
    
    t = -point[2] / rotated_vector[2]
    
    # Calculer le point sur Z = 0
    point_z0 = point + t * rotated_vector
    
    return point_z0

def visualize_points_with_z0_projections(points, quaternions):
    """
    Visualise des points dans un espace 3D avec leurs projections sur Z = 0.
    
    Parameters:
    points (np.array): Tableau de points de forme (n, 3)
    quaternions (np.array): Tableau de quaternions de forme (n, 4)
    """
    # Créer une figure 3D
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Vérifier que les tableaux ont la même longueur
    assert len(points) == len(quaternions), "Le nombre de points et de quaternions doit être identique"
    
    # Couleurs pour les différents points
    colors = plt.cm.viridis(np.linspace(0, 1, len(points)))
    
    # Liste pour stocker les points projetés
    z0_projections = []
    
    # Visualiser chaque point et sa projection
    for i, (point, quaternion) in enumerate(zip(points, quaternions)):
        # Convertir le quaternion en matrice de rotation
        rotation = R.from_quat(quaternion)
        
        # Calculer la direction du vecteur
        direction = rotation.apply([0, 0, 1])
        
        # Calculer le point projeté sur Z = 0
        point_z0 = calculate_z0_point_from_quaternion(point, quaternion)
        z0_projections.append(point_z0)
        
        # Tracer le point original
        ax.scatter(point[0], point[1], point[2], color=colors[i], s=100)
        
        # Tracer le point projeté sur Z = 0
        ax.scatter(point_z0[0], point_z0[1], 0, color='green', s=200, marker='x')
        
        # Tracer la ligne entre le point original et sa projection
        ax.plot([point[0], point_z0[0]], 
                [point[1], point_z0[1]], 
                [point[2], 0], 
                color=colors[i], linestyle='--', alpha=0.5)
        
        # Tracer la direction du quaternion
        ax.quiver(point[0], point[1], point[2], 
                  direction[0], direction[1], direction[2], 
                  color=colors[i], 
                  length=0.5, 
                  arrow_length_ratio=0.1)

    print("puntos: ", z0_projections)
    p1, p2, p3 = z0_projections
    distp1p2 = np.linalg.norm(p1 - p2)
    distp1p3 = np.linalg.norm(p1 - p3)
    distp2p3 = np.linalg.norm(p2 - p3)
    print("distp1p2: ", distp1p2)
    print("distp1p3: ", distp1p3)
    print("distp2p3: ", distp2p3)


    # Tracer le plan Z = 0
    xx, yy = np.meshgrid(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 2),
                         np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 2))
    ax.plot_surface(xx, yy, np.zeros_like(xx), color='gray', alpha=0.2)

    
    # Configurer les labels et le titre
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Projection des Points sur Z=0 selon l\'Orientation')
    
    # Ajuster les limites pour une meilleure vue
    ax.set_box_aspect((1,1,1))
    
    plt.tight_layout()
    plt.show()


# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple de points et de quaternions
    example_points = np.array([
        [0.172559, 0.425112, 0.206797],    # Point 1
        [0.109986, 0.201156, 0.206593],    # Point 2
        [-0.165158, 0.387674, 0.206637],   # Point 3
    ])
    
    example_quaternions = np.array([
        [-0.296215, 0.889204, -0.0385748, -0.346533],
        [0.353087, 0.868182, 0.196004, -0.288395],
        [0.296254, 0.889193, 0.0385806, 0.346528],
    ])
    
    ([0.172559, 0.425112, 0.206797], [-0.296215, 0.889204, -0.0385748, -0.346533])
    ([0.109986, 0.201156, 0.206593], [0.353087, 0.868182, 0.196004, -0.288395])
    ([-0.165158, 0.387674, 0.206637], [0.296254, 0.889193, 0.0385806, 0.346528])
    

    # Visualiser les points
    visualize_points_with_z0_projections(example_points, example_quaternions)

