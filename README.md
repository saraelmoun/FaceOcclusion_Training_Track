


<div align="center">

# 🎭 Face Occlusion Estimation 



<br/>

![Track](https://img.shields.io/badge/Track-Model%20Training-8A2BE2?style=for-the-badge)
![Task](https://img.shields.io/badge/Task-Régression-blue?style=for-the-badge)

</div>

---

## Contexte

La reconnaissance faciale repose sur la visibilité des traits du visage. En conditions réelles, ceux-ci
sont fréquemment masqués masque sanitaire, main, lunettes, cheveux, ou tout autre objet. Mesurer
**quelle proportion d'un visage est occultée** est donc une étape clé pour estimer la fiabilité d'un
système biométrique. C'est le problème posé par ce challenge, fourni par **IDEMIA**.

## Problématique



<div align="center">

</br>

| Entrée | Sortie | Contrainte |
|:---:|:---:|:---:|
| Image visage `224×224` | Score d'occlusion `[0, 1]` | Équité Femmes / Hommes |


<br/>

<img src="images/faceocclusionmeme.jpeg" alt="Exemples d'occlusions de visage" width="460"/>



</div>

---

## La métrique d'évaluation

L'erreur est une **MSE pondérée** qui pénalise davantage les fortes occlusions, puis on **moyenne par genre** avec une **pénalité de disparité** :

```math
\mathrm{Err} = \frac{\sum_i w_i\,(p_i - GT_i)^2}{\sum_i w_i}, \qquad w_i = \frac{1}{30} + GT_i
```

```math
\mathrm{Score} = \frac{\mathrm{Err}_F + \mathrm{Err}_M}{2} \;+\; \bigl|\,\mathrm{Err}_F - \mathrm{Err}_M\,\bigr|
```


---

## Reproduire les résultats

```bash
# 1. Cloner le dépôt
git clone <url-du-repo>
cd FaceOcclusion_TrainingTrack

# 2. Environnement
pip install -r requirements.txt

# 3. Télécharger les images dans DATA_ROOT (voir lien partage.imt.fr)
#    puis ajuster DATA_ROOT dans les notebooks

# 4. Explorer les données
jupyter notebook EDA.ipynb

# 5. Entraîner le modèle puis générer les prédictions de test
#    (seed fixée pour la reproductibilité)
```

> Le pipeline d'entraînement produit les poids du modèle et `test_predictions.csv` au format requis pour la soumission.

---

## 🥉 Résultats

<div align="center">

<img src="images/leaderboard.png" alt="Classement du data challenge" width="600"/>

<sub><i>Classement interim — Data Challenge IDEMIA × Télécom Paris.</i></sub>

</div>

---

<div align="center">

`FacePredict Team` — `Group 11`


<table border="0"><tr>
<td align="center" valign="middle" width="280"><img src="images/Idemia.png" alt="IDEMIA" width="200"/></td>
<td align="center" valign="middle" width="280"><img src="images/TelecomParis.png" alt="Télécom Paris" width="150"/></td>
</tr></table> 

*Data Challenge IDEMIA × Télécom Paris · Institut Polytechnique de Paris*

</div>
