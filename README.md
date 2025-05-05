# clothing-classifier
Classificateur d'images basé sur un réseau de neurones convolutif (CNN) pour identifier des articles de mode à partir du jeu de données Fashion MNIST. Projet réalisé avec TensorFlow/Keras dans le cadre d'une préparation à une alternance en IA.

 # Classificateur d'articles de vêtements (TensorFlow + CNN)

Ce projet implémente un réseau de neurones convolutif (CNN) avec TensorFlow/Keras pour classer des images en niveaux de gris d’articles de mode issues du jeu de données [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist).

Il atteint une **précision supérieure à 90 %** sur les données de test en moins de 5 époques.

---

## Architecture du modèle

```text
Entrée : images 28x28 en niveaux de gris

1. Conv2D (32 filtres, noyau 3x3, activation ReLU)
2. MaxPooling2D (2x2)
3. Conv2D (64 filtres, noyau 3x3, activation ReLU)
4. MaxPooling2D (2x2)
5. Flatten
6. Dense (128 neurones, activation ReLU)
7. Dense (10 neurones, activation Softmax)

