# Adult Income Classifier — pipeline MLOps de bout en bout

[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)](.github/workflows/ci.yml)

Pipeline de classification tabulaire reproductible (scikit-learn `Pipeline`
+ `ColumnTransformer`), tuning par validation croisée, **suivi et
gouvernance complets dans MLflow** (tracking, registre, promotion
champion/challenger), API FastAPI conteneurisée, et publication optionnelle
du modèle champion sur le Hugging Face Hub avec une démo qui tourne dans le
navigateur.

Projet réalisé dans le cadre du cours MLOps. Le dataset
et le squelette de départ suivent la consigne du cours ; l'implémentation,
les choix de conception et la gouvernance de modèle (seuil optimisé,
promotion conditionnelle, contrat de version, détection de dérive) sont
originaux.

```
make data ──▶ make validate ──▶ make train ──▶ make promote ──▶ make export ──▶ make serve
   CSV UCI       schéma OK        GridSearchCV     champion ?      artifacts/        API
                                  + seuil OOF       (registre)      model.joblib   /predict /health
                                  MLflow run        alias champion  model_meta.json
```

---

### Le dataset : UCI Adult Income

Extrait du recensement américain de 1994. Tâche : prédire si une personne
gagne plus de 50 000 $/an. Choisi pour trois raisons pédagogiques : un
vrai déséquilibre de classes (~24 % de positifs, donc le seuil de décision
compte), des colonnes catégorielles à forte cardinalité (`native_country`,
41 modalités) qui obligent à gérer les catégories rares proprement, et des
attributs sensibles (`sex`, `race`) qui permettent de vérifier l'équité du
modèle plutôt que de se contenter d'une métrique globale.

| | |
|---|---|
| Lignes | 48 842 (`adult.data` + `adult.test`, split contrôlé par la config) |
| Features numériques | `age`, `education_num`, `hours_per_week` |
| Features numériques asymétriques | `capital_gain`, `capital_loss` (log1p) |
| Features catégorielles | `workclass`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country` |
| Classe positive | ~24 % (`>50K`) |

Particularités du fichier brut gérées dans `src/data.py` : pas d'en-tête
(colonnes fournies par la config), un espace devant chaque valeur, `?` pour
les valeurs manquantes, et un point final parasite sur la cible du fichier
de test (`>50K.`).

**Deux colonnes sont retirées explicitement** (`src/train.py`, section
`features.drop` de la config) :
- `fnlwgt` : poids d'échantillonnage du recensement, pas une caractéristique
  de la personne — l'inclure ferait apprendre au modèle un artefact du plan
  de sondage plutôt qu'un vrai signal.
- `education` : redondante avec `education_num`, déjà ordinale.

### Le seuil de décision n'est pas 0,5

Sur un dataset à 24 % de positifs, un seuil à 0,5 favorise mécaniquement la
classe majoritaire. Après le `GridSearchCV`, on récupère des probabilités
**out-of-fold** (`cross_val_predict`, jamais sur le split de test) et on
choisit le seuil qui maximise le F1 sur ces probabilités
(`src/metrics.py::best_threshold_oof`). Ce seuil est loggé dans MLflow,
stocké dans `model_meta.json`, et **consommé partout de la même façon** :
`evaluate.py`, `predict.py` et l'API ne réimplémentent jamais leur propre
logique de seuillage.

### Gouvernance de modèle : champion / challenger

Chaque `make train` enregistre une nouvelle version dans le registre sous
l'alias `challenger` — il ne remplace jamais automatiquement le modèle
servi. `make promote` compare le `test_roc_auc` du challenger à celui du
champion courant (même split de test, même seed) et ne déplace l'alias
`champion` que si le gain dépasse `mlflow.promotion.min_improvement`
(configurable). Conséquence directe : **l'API ne sert jamais un modèle qui
n'a pas explicitement gagné cette comparaison**, puisque `make export` ne
tire que le champion. Un modèle moins bon peut être entraîné sans risque —
il reste `challenger` tant qu'il n'a pas prouvé qu'il est meilleur.

### `model_meta.json` : le contrat entre le registre et le service

MLflow est lourd (tracking server, base de runs) et n'a pas sa place dans
l'image de service. `export.py` exporte le champion vers un artefact léger
(`model.joblib`) accompagné de `model_meta.json` — version, `run_id`,
seuil, type de modèle. L'API charge uniquement ces deux fichiers : elle
sait exactement quelle version elle sert et peut le déclarer dans chaque
réponse (`/predict`, `/health`), sans jamais interroger MLflow au moment
de l'inférence.

### Validation avant entraînement, pas après

`src/validate.py` vérifie le schéma, le volume de données, le taux de
valeurs manquantes et le taux de classe positive **avant** de lancer quoi
que ce soit de coûteux. Un CSV corrompu échoue en quelques millisecondes,
pas après plusieurs minutes de `GridSearchCV`.

---

## Structure du repo

```
adult-income-mlops/
├─ configs/
│  └─ config.yaml            # données, features, grille, CV, MLflow, promotion — seul point de config
├─ src/
│  ├─ data.py                 # téléchargement UCI -> data/raw.csv
│  ├─ validate.py              # validation de schéma avant entraînement
│  ├─ pipeline.py               # ColumnTransformer + modèle (Pipeline sklearn)
│  ├─ metrics.py                 # métriques + seuil de décision optimal (F1, out-of-fold)
│  ├─ train.py                    # GridSearchCV + MLflow (tracking + registre, alias `challenger`)
│  ├─ promote.py                   # promotion champion/challenger conditionnelle
│  ├─ evaluate.py                   # évaluation détaillée (ROC/PR/confusion) + logs MLflow
│  ├─ export.py                      # registre -> artifacts/ (model.joblib + model_meta.json)
│  ├─ predict.py                      # inférence batch CSV -> CSV
│  ├─ app.py                           # API FastAPI (/predict, /health)
│  ├─ export_onnx.py                    # export ONNX validé contre scikit-learn (pour le Space HF)
│  ├─ publish.py                         # publication du champion sur le Hugging Face Hub
│  ├─ deploy_space.py                     # déploiement du Space HF statique
│  └─ utils.py                             # config, logging, hash de données, IO JSON
├─ tests/
│  ├─ synthetic_data.py       # génère un CSV synthétique (même schéma) — aucune dépendance réseau
│  ├─ test_pipeline.py         # pipeline + validation (NaN, catégories inconnues, données dégénérées)
│  ├─ test_utils.py             # métriques, seuil OOF, hash, IO
│  ├─ test_api.py                # API sans modèle chargé -> 503
│  └─ test_e2e.py                 # cycle complet train -> promote -> export sur registre temporaire
├─ deploy/
│  └─ space/                  # Space HF statique : index.html + app.js (inférence ONNX côté navigateur)
├─ .github/workflows/ci.yml   # lint + tests + smoke-train + build Docker + déploiement HF sur tag v*
├─ Makefile · Dockerfile · requirements.txt · pyproject.toml · .env.example
└─ README.md
```

---

## Quickstart (100 % local)

Prérequis : Python 3.12, Docker (optionnel, pour la conteneurisation).

```bash
make init                    # venv + dépendances
cp .env.example .env
make data                    # télécharge adult.data + adult.test -> data/raw.csv
make validate                # vérifie le schéma avant d'aller plus loin
make train                    # GridSearchCV + MLflow (params, métriques, artefacts, registre)
make promote                   # premier modèle : promu champion par défaut
make evaluate                   # courbes ROC/PR/confusion, predictions.csv
make export                      # registre -> artifacts/model.joblib + model_meta.json
make ui                           # MLflow UI sur http://127.0.0.1:5001
make test                          # pytest — 14 tests (unitaires, API, bout-en-bout)
make lint                           # ruff check + ruff format --check
```

Entraîner un autre type de modèle et laisser `promote` décider :

```bash
# modifier configs/config.yaml -> model.type: random_forest, puis :
make train
make promote        # ne remplace le champion que si strictement meilleur
```

### Servir le modèle

```bash
make serve
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d @examples/person.json
# {"label":">50K","score":0.87,"threshold":0.41,"model_name":"AdultIncomeClassifier",
#  "model_version":"1","run_id":"...","latency_ms":3.2}
curl localhost:8000/health   # version, run_id, seuil du modèle actuellement servi
```

### Inférence batch

```bash
make predict INPUT=data/new.csv OUTPUT=artifacts/scored.csv
```

### Docker

```bash
make docker-build
make docker-serve    # API seule, sert artifacts/model.joblib (monté en volume)
```

---

## Ce que MLflow trace

**Run d'entraînement** (`train.py`) :
- Lignée des données : hash SHA-256 du CSV, nombre de lignes, taux de classe
  positive — deux runs sur des données différentes sont distinguables.
- Meilleurs hyperparamètres (`GridSearchCV`), `decision_threshold`.
- Métriques CV (`cv_best_roc_auc`) et test tenu à l'écart (`test_roc_auc`,
  `test_average_precision`, `test_brier`, `test_f1`, `test_f1_at_0.5`).
- Modèle loggé avec signature d'entrée/sortie, enregistré dans le registre
  sous l'alias `challenger`.

**Run `promote-vN`** : la décision de promotion (comparaison, scores,
promu ou non) est elle-même tracée comme un run.

**Run `evaluate`** : courbes ROC / PR / matrice de confusion, export des
prédictions.

---

## API

| Endpoint | Description |
|---|---|
| `POST /predict` | Score + label + seuil + version du modèle servi |
| `GET /health` | Version, `run_id`, seuil, statut du modèle chargé |

Chaque prédiction est traçable jusqu'au run MLflow qui a produit le modèle
via le `run_id` retourné dans la réponse.

---

## Publication Hugging Face (optionnelle, pilotée par la CI)

GitHub reste la source de vérité ; Hugging Face héberge uniquement ce que
la CI a produit et validé.

```
GitHub (code, tests, CI) ──▶ tag v* ──▶ export ONNX (validé vs sklearn)
                                            │
                                            ▼
                              HF Hub, dépôt modèle (model.joblib + model.onnx
                              + model_meta.json + model card, tag vN)
                                            │
                                            ▼
                              HF Space statique (index.html + app.js)
                              télécharge model.onnx et prédit dans le
                              navigateur — aucun serveur à héberger pour la démo
```

- **`make export-onnx`** convertit le champion en ONNX et **valide l'export
  contre scikit-learn** sur un échantillon (écart de probabilité toléré :
  1e-4). Le pipeline utilise `log1p` sur `capital_gain`/`capital_loss` :
  cette transformation n'a pas de convertisseur ONNX natif dans ce projet,
  elle est donc réappliquée côté client en JavaScript
  (`deploy/space/app.js`) d'après les colonnes listées dans
  `preprocess.json`, généré au même moment que le modèle pour rester
  synchronisé.
- **`make publish-model`** pousse le champion (joblib + ONNX + métadonnées
  + model card générée) sur un dépôt de modèle HF, taggé `v<version du
  registre>`. Refuse toute version qui n'est pas l'alias `champion`.
- **`make deploy-space`** déploie `deploy/space/` (page statique) avec un
  `config.json` généré au déploiement, qui pointe le dépôt de modèle et la
  révision à charger (`deploy/space.env`, `main` par défaut).

Mise en place (une fois) : secret `HF_TOKEN` dans l'environment GitHub
`huggingface`, variables de repo `HF_MODEL_REPO` / `HF_SPACE_REPO`. Ensuite,
chaque tag `v*` poussé déclenche automatiquement le job `deploy-huggingface`
de la CI (voir `.github/workflows/ci.yml`).

---

## CI (GitHub Actions)

- **test** : `ruff check` + `ruff format --check` + `pytest` (14 tests, dont
  le cycle bout-en-bout sur données synthétiques et registre SQLite
  temporaire — aucune dépendance réseau).
- **smoke-train** : cycle complet sur le vrai dataset UCI
  (`data → validate → train → promote → evaluate → export`), artefacts
  uploadés.
- **docker** : build de l'image de service.
- **deploy-huggingface** : uniquement sur un tag `v*` — reconstitue le
  champion, exporte en ONNX, publie sur le Hugging Face Hub, déploie le
  Space statique.

---

## Choix techniques (résumé)

- **Tout passe par `configs/config.yaml`** : changer de modèle ou de
  features ne touche jamais au code.
- **Validation avant entraînement**, pour échouer vite sur des données
  corrompues.
- **Seuil choisi en out-of-fold**, jamais sur le split de test.
- **Alias `challenger`/`champion`** plutôt que promotion automatique :
  aucun modèle n'atteint la production sans avoir battu l'existant.
- **`model_meta.json`** comme contrat unique entre le registre MLflow et
  tout ce qui sert le modèle (API, export batch, Space HF).
- **Export ONNX validé numériquement**, pas seulement "converti sans
  erreur" — la validation compare les probabilités, pas juste l'absence
  d'exception.


## Lien vers HF

- https://huggingface.co/spaces/mensahkodjosamuel/adult-income-classifier-demo


