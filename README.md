# Projet Deep Learning — EMSI Casablanca 2025–2026

**Module** : Deep Learning | **Filiere** : Informatique | **Auteur** : Adam GHZAOUNI  
**Etablissement** : EMSI Mouley Youssef, Casablanca

---

## Idee generale du projet

Ce projet explore trois familles architecturales fondamentales du deep learning, chacune adaptee a une modalite de donnees differente :

| Partie | Architecture | Donnees | Tache |
|--------|--------------|---------|-------|
| I | MLP (Perceptron Multicouche) | MedTab-BC — dataset tabulaire clinique synthetique (569 entrees, 30 biomarqueurs) | Classification binaire : tumeur maligne vs benigne |
| II | CNN (Reseau Convolutionnel) | SynthXRay-64 — radiographies pulmonaires generees procedurallement (1600 images 64x64) | Detection de pneumonie (binaire) |
| III | RNN / LSTM / GRU + Seq2Seq | MTSamples — 4 999 transcriptions medicales reelles issues du projet Tatoeba (40 specialites) | Classification de specialite + Resume automatique de comptes-rendus |

L'ensemble des experiences est conduit sous PyTorch avec une graine aleatoire fixee (SEED=42) pour garantir la reproductibilite. Chaque partie articule fondements theoriques, implementation commentee, experimentation comparative et analyse critique.

---

## Datasets

### MedTab-BC (Partie I)

Dataset tabulaire clinique simule a partir de la distribution statistique du Breast Cancer Wisconsin original. Il contient 569 entrees et 30 biomarqueurs numeriques (rayon moyen, texture, perimetre, aire, concavite, symetrie, etc.) organises en deux classes : tumeur maligne (212 cas) et tumeur benigne (357 cas). La generation est reproductible via SEED=42 et sklearn. Aucun telechargement requis.

### SynthXRay-64 (Partie II)

Dataset d'imagerie pulmonaire genere procedurallement. Chaque image (64x64, canal unique, niveaux de gris normalises) est construite par superposition de geometrie elliptique, de masques d'opacite aleatoires et d'un bruit gaussien calibre. La classe Normal produit des poumons clairs ; la classe Pneumonie y ajoute entre 3 et 8 foyers d'opacite de taille et de position aleatoires (r entre 4 et 12 pixels). Le dataset compte 1600 images equilibrees (800 par classe). Aucun telechargement requis.

### MTSamples (Partie III)

Dataset de transcriptions medicales reelles extrait du site MTSamples.com, distribue publiquement via GitHub. Il contient 4 999 comptes-rendus cliniques couvrant 40 specialites medicales (Chirurgie, Cardiologie, Neurologie, Radiologie, Orthopedie, Medecine Generale, etc.) avec pour chaque entree : la specialite, le nom de l'examen, la transcription complete et les mots-cles associes. Les 10 specialites les plus representees totalisent 3 657 rapports.

Source : github.com/salgadev/medical-nlp — `data/mtsamples.csv`  
Licence : domaine public (donnees de demonstration medicale)

---

## Resultats cles

### Partie I — MLP sur MedTab-BC

- Accuracy sur le test set : 97.08%
- Comparaison de 3 strategies d'initialisation : Gaussienne, Constante, Xavier (Glorot)
- Xavier produit une train loss finale 5.9 fois inferieure a l'initialisation constante
- F1-score pondere : 0.97

### Partie II — CNN sur SynthXRay-64

- 85.4% accuracy sur le test set — avec 2.3 fois moins de parametres que le MLP applique aux pixels bruts
- Etude d'ablation sur 5 facteurs : padding, pooling, nombre de filtres, stride, type de pooling
- Le nombre de filtres est le levier dominant (delta = +3.2% entre filters=16 et filters=64)
- Visualisation des cartes de caracteristiques par bloc convolutif via `get_feature_maps()`

### Partie III — RNN/LSTM/GRU + Seq2Seq sur MTSamples

- GRU : perplexite 3.34 (meilleure), LSTM : 3.35, RNN : 3.39
- Systeme Seq2Seq encodeur-decodeur LSTM avec teacher forcing (ratio=0.5)
- Score BLEU-1 moyen : Beam Search (k=3) superieur de +16% au decodage glouton
- Stabilisation du gradient clipping demontree experimentalement (clip=1.0 vs sans clipping)

---

## Structure du depot

```
medical-deep-learning/
│
├── notebooks/
│   ├── 01_MLP_BreastCancer.ipynb         # Partie I : MLP + MedTab-BC
│   ├── 02_CNN_ChestXRay.ipynb            # Partie II : CNN + SynthXRay-64
│   └── 03_LSTM_MedicalReports.ipynb      # Partie III : RNN/LSTM/GRU + MTSamples
│
├── data/
│   └── mtsamples/
│       └── mtsamples.csv                 # 4 999 transcriptions medicales (40 specialites)
│
├── requirements.txt                      # Dependances Python
└── README.md
```

---

## Contenu detaille des notebooks

### 01_MLP_BreastCancer.ipynb — Partie I : MLP sur MedTab-BC

- Setup reproductibilite (SEED=42, detection device GPU/CPU)
- Generation et exploration du dataset MedTab-BC (569 echantillons, 30 biomarqueurs)
- Preparation sans data leakage : split stratifie 70/15/15, StandardScaler fit sur train uniquement
- Implementation MLP en deux variantes : `nn.Sequential` et classe personnalisee `BreastCancerMLP(nn.Module)`
- Inspection des parametres : `named_parameters()`, `state_dict()`
- Comparaison de 8 configurations : 3 activations (ReLU, Tanh, LeakyReLU), 3 profondeurs, 3 initialisations
- Evaluation finale : matrice de confusion, courbe ROC, rapport de classification sklearn

### 02_CNN_ChestXRay.ipynb — Partie II : CNN sur SynthXRay-64

- Generation procedurale du dataset SynthXRay-64 (1600 images 64x64 avec `SyntheticChestXRay`)
- Implementation manuelle validee numeriquement : `corr2d()`, `max_pool()`, `avg_pool()`
- Visualisation de l'effet de differents noyaux (detection de bords, flou gaussien) sur les radiographies
- Architecture `PneumoniaCNN` : 3 blocs Conv-BN-ReLU-Pool, classifieur FC avec dropout
- Entrainement 60 epoques avec CosineAnnealingLR et early stopping (patience=12)
- Etude d'ablation systematique sur 6 configurations (padding, stride, pooling, nombre de filtres)
- Visualisation des feature maps par bloc et des filtres du premier bloc via `get_feature_maps()`
- Comparaison MLP vs CNN : accuracy, nombre de parametres, courbes ROC superposees

### 03_LSTM_MedicalReports.ipynb — Partie III : RNN/LSTM/GRU + Seq2Seq sur MTSamples

- Chargement et preprocessing de `data/mtsamples/mtsamples.csv` (nettoyage, filtrage des specialites majeures)
- Construction du vocabulaire medical avec tokens speciaux PAD, UNK, SOS, EOS (classe `Vocabulary`)
- Implementation des 3 architectures recurrentes : `nn.RNN`, `nn.LSTM`, `nn.GRU` (memes hyperparametres)
- Entrainement comparatif — metriques : loss, accuracy, perplexite, norme moyenne du gradient par epoque
- Illustration experimentale du gradient clipping (sans clipping, clip=1.0, clip=5.0)
- Architecture Seq2Seq : `Encoder` LSTM + `Decoder` LSTM avec teacher forcing (ratio=0.5)
- Decodage glouton (`greedy_decode`) et Beam Search (`beam_search`, k=2, 3, 5)
- Evaluation BLEU-1 sur le jeu de validation, analyse qualitative des resumes generes

---

## Installation et utilisation

### Prerequis

- Python 3.11 ou superieur
- GPU recommande (CUDA) mais CPU supporte

### Installation des dependances

```bash
pip install -r requirements.txt
```

### Lancer les notebooks

```bash
jupyter notebook notebooks/
```

Ouvrir dans l'ordre :

1. `01_MLP_BreastCancer.ipynb` — Partie I
2. `02_CNN_ChestXRay.ipynb` — Partie II
3. `03_LSTM_MedicalReports.ipynb` — Partie III

Le dataset MTSamples est deja present dans `data/mtsamples/mtsamples.csv`. Les datasets MedTab-BC et SynthXRay-64 sont generes automatiquement au lancement des notebooks correspondants.

---

## Dependances principales

| Librairie | Usage |
|-----------|-------|
| torch | Modeles, boucles d'entrainement, autograd |
| torchvision | Transforms et utilitaires image |
| scikit-learn | Metriques, StandardScaler, train_test_split |
| matplotlib, seaborn | Courbes, matrices de confusion, feature maps |
| numpy, pandas | Generation et manipulation des datasets |

---

## References

- Goodfellow, Bengio, Courville (2016). *Deep Learning*. MIT Press.
- LeCun et al. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*.
- Hochreiter & Schmidhuber (1997). Long short-term memory. *Neural Computation*.
- Cho et al. (2014). Learning phrase representations using RNN encoder-decoder. *EMNLP*.
- Glorot & Bengio (2010). Understanding the difficulty of training deep feedforward networks. *AISTATS*.
- Paszke et al. (2019). PyTorch: An imperative style high-performance deep learning library. *NeurIPS*.

---

Annee universitaire 2025–2026 — EMSI Mouley Youssef, Casablanca
