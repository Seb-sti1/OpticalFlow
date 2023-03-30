import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Parameters
file_path = '../data/data.csv'
model_save_path = '../models/'

# Préparation des données
data = pd.read_csv(file_path, skipinitialspace=True)

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# GridSearch SVM
svc = SVC(shrinking=True, max_iter=1000000)  # max_iter = 1000000 pour limiter les non-convergences de l'optimiseur
parameter_dict = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': np.logspace(0.01, 10, 20), 'degree': range(2, 8),
                  'gamma': ['scale', 'auto'], 'decision_function_shape': ['ovo', 'ovr']}

grid_search = GridSearchCV(svc, parameter_dict, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Afficher les meilleurs hyperparamètres trouvés par la recherche sur grille
print("Meilleurs hyperparamètres SVC :", grid_search.best_params_)
# Afficher le meilleur score obtenu lors de la validation croisée
print("Meilleur score de validation croisée SVC :", grid_search.best_score_)

# Évaluer le modèle avec les meilleurs hyperparamètres sur l'ensemble de test
best_svc = grid_search.best_estimator_
test_score = best_svc.score(X_test, y_test)
print("Score sur l'ensemble de test :", test_score)

# Sauvegarder le modèle entraîné dans un fichier
joblib.dump(best_svc, model_save_path + 'svc_plan.joblib')
print(f"svc model saved as svc_plan.joblib")

#  GridSearch kNN
knn = KNeighborsClassifier()
parameter_dict = {'n_neighbors': [1, 3, 5, 7, 9], 'weights': ['uniform', 'distance'],
                  'metric': ['euclidean', 'manhattan', 'minkowski', 'cosine']}

# Créer une instance de GridSearchCV avec le modèle k-NN et les hyperparamètres
grid_search = GridSearchCV(knn, parameter_dict, scoring='accuracy')

# Ajuster le modèle à l'aide de la recherche sur grille et des données d'entraînement
grid_search.fit(X_train, y_train)

# Afficher les meilleurs hyperparamètres trouvés par la recherche sur grille
print("Meilleurs hyperparamètres k-NN :", grid_search.best_params_)

# Afficher le meilleur score obtenu lors de la validation croisée
print("Meilleur score de validation croisée k-NN :", grid_search.best_score_)

# Évaluer le modèle avec les meilleurs hyperparamètres sur l'ensemble de test
best_knn = grid_search.best_estimator_
test_score = best_knn.score(X_test, y_test)
print("Score sur l'ensemble de test :", test_score)

# Sauvegarder le modèle entraîné dans un fichier
joblib.dump(best_knn, model_save_path + 'knn_plan.joblib')
print(f"k-NN model saved as knn_plan.joblib")
