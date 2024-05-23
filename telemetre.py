class telemetre:
    """
    Classe permettant de calculer la position d'un objet en 3D
    x, y, z : coordonnées de l'objet
    theta: angle de tangage autour de l'axe x
    psi: angle de roulis autour de l'axe z
    mesure: distance entre l'objet et le telemetre
    """

    def __init__(self, x, y, z, theta, psi, mesure):
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta
        self.psi = psi
        self.mesure = mesure
