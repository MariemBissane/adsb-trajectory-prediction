# ✈️ Prédiction de Trajectoires d'Avions à partir de Données ADS-B

##  Description

Ce projet vise à prédire la trajectoire future des avions à partir de données ADS-B issues du réseau OpenSky.
L'objectif est d'exploiter des séries temporelles réelles, bruitées et irrégulières afin de construire des modèles de Machine Learning et Deep Learning capables d’anticiper le comportement des avions.

---

## Dataset

* Source : OpenSky Network Historical Data
* Lien : https://opensky-network.org/datasets/#states/

 Stratégie :

* Utiliser un **échantillon réduit** pour le prototypage
* Étendre ensuite à un dataset plus large pour améliorer les performances

---

##  Objectifs

Prédire la trajectoire future d’un avion sur un horizon de **2 à 10 minutes** selon deux approches :

### Régression

* Prédire les **N prochaines positions** :

  * Latitude
  * Longitude
  * Altitude

###  Classification

* Prédire l’**intention de vol** :

  * Approche
  * Montée
  * Virage
  * Maintien

---

#  PARTIE 1 — Machine Learning

##  Prétraitement des données

* Resampling temporel :

  * Uniformisation (ex : 1 point toutes les 5s ou 10s)
* Filtrage du bruit
* Normalisation des variables

---

##  Feature Engineering

### Variables physiques

* Ground speed
* Vertical rate
* Acceleration
* Turn rate

### Variables géométriques

* Distance à la destination
* Angle de trajectoire

### Variables statistiques

* Rolling mean / std (fenêtres glissantes)

### Variables métier

* Phase de vol estimée

---

##  Réduction de dimension

* PCA sur les features dérivées

---

##  Modèles utilisés

* Random Forest (régression multi-sortie)
* Gradient Boosting
* SVM (kernel RBF) pour classification
* Baseline : Arbre de décision

---

##  Évaluation

### Régression

* RMSE / MAE (lat, lon, altitude)
* Distance moyenne de déviation

### Classification

* Accuracy
* F1-score

---

#  PARTIE 2 — Deep Learning

##  Architectures

### Réseaux récurrents

* LSTM
* GRU

Entrée :

```
[lat, lon, alt, vx, vy, vz, heading]
```

### Réseaux convolutifs temporels

* TCN (Temporal Convolutional Network)

---

##  Livrables Deep Learning

* Visualisation trajectoire réelle vs prédite (Folium / Plotly)
* Courbes d’erreur selon horizon (1min, 3min, 5min)
* Analyse de l’impact du resampling

---

#  Optimisation & Entraînement

##  Learning Rate

* Scheduler (ex : ReduceLROnPlateau)

##  Stabilité & Généralisation

* Étude de la taille des minibatchs
* Utilisation du Dropout

##  Contrôle de l’entraînement

* EarlyStopping
* GridSearchCV / Optuna

---

#  Analyse de Performance

* Courbes d’apprentissage (Train vs Validation)
* Comparaison avant/après tuning
* Analyse des résidus (régression)
* Matrice de confusion / ROC-AUC (classification)

---

#  Structure du projet

( pas encore définit )
---

#  Livrables

*  Notebook Machine Learning
*  Notebook Deep Learning
*  Pipeline automatique
*  Rapport technique

---

#  Technologies utilisées

* Python
* Pandas / NumPy
* Scikit-learn
* PyTorch / TensorFlow
* Matplotlib / Seaborn / Plotly

---

#  Conclusion

Ce projet permet de manipuler des données réelles complexes et d’explorer des approches avancées en Machine Learning et Deep Learning pour la prédiction de trajectoires aériennes.

---

##  Auteur

* Projet académique — Génie Informatique
