# Programme Telemetre 
## Description
Le programme permet d'utiliser une camera oak-d lite pour determiner la position d'un ArUco marker dans l'espaces.

## Installation
Pour installer le programme, il suffit de cloner le repertoire git et d'installer les dependances python.
```bash
git clone
pip install -r requirements.txt
```

## Utilisation
Pour utiliser le programme, il suffit de lancer le script python.
```bash
python main.py
```
à l'approche d'un ArUco marker, la position de celui-ci sera affichée sur l'image. Les images des arUco marker sont disponibles dans le repertoire `aruco_marker` et sont tirées de la librairie 'aruco.DICT_4X4_1000'.