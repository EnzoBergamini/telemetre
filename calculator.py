from telemetre import telemetre


class calculator:
    @staticmethod
    def distance(telemetre1, telemetre2):
        """
        Calcule la distance entre deux telemetres
        """
        return (
            (telemetre1.x - telemetre2.x) ** 2
            + (telemetre1.y - telemetre2.y) ** 2
            + (telemetre1.z - telemetre2.z) ** 2
        ) ** 0.5
