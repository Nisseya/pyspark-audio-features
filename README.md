## Quickstart

```bash
uv sync
``` 

pour récupérer les dépendances


## Download le dataset:

```sh
uv run python -m scripts.download
```

Cette commande ne télécharge que 5% des données, soit un peu plus de 10 go.

pour télécharger le dataset en entier, il faut faire

```sh
uv run python -m scripts.download --full
```

Celui ci en fait 250, donc ce sera pour le processing complet.

## Convertir en wav 
On fait ca pour pouvoir extraire les features etc directement par la suite, car mp3 est compressé donc pas analysable

pour faire ca il faut ffmpeg:

### Linux: (ubuntu/debian)

```sh
sudo apt update
sudo apt install ffmpeg -y
```

### Windows:
Débrouillez vous

### Mac (Ilya):
Débrouille toi

Après la commande c
```sh
uv run python -m scripts.convert
```

C'est un peu long, j'ai optimisé comme j'ai pu

Vous pouvez augmenter les workers selon votre CPU si vous voulez.