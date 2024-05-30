from telemetre import telemetre
import numpy as np


class calculator:
    @staticmethod
    def distance(telemetre1, telemetre2):
        """
        Calcule la distance entre deux telemetres
        """
        return np.sqrt(
            (
                telemetre1.x * np.cos(telemetre1.psi)
                - telemetre2.x * np.cos(telemetre2.psi)
            )
            ** 2
            + (
                (telemetre1.y + telemetre1.z + telemetre1.mesure)
                * np.sin(telemetre1.theta)
                - (telemetre2.y + telemetre2.z + telemetre2.mesure)
                * np.sin(telemetre2.theta)
            )
            ** 2
            + (
                telemetre1.x * np.sin(telemetre1.psi)
                + (telemetre1.y + telemetre1.z + telemetre1.mesure)
                * np.cos(telemetre1.theta)
                - telemetre2.x * np.sin(telemetre2.psi)
                - (telemetre2.y + telemetre2.z + telemetre2.mesure)
                * np.cos(telemetre2.theta)
            )
            ** 2
        )
