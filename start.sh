#!/bin/bash

# Script de démarrage automatique pour l'outil de traitement d'images
# Ce script démarre automatiquement le backend et le frontend

echo "Démarrage de l'outil de traitement d'images..."

# Vérifier que nous sommes dans le bon répertoire
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "Erreur: Ce script doit être exécuté depuis le répertoire racine du projet"
    echo "   Assurez-vous d'être dans le répertoire image-processing-tool/"
    exit 1
fi

# Fonction pour nettoyer les processus en arrière-plan
cleanup() {
    echo "Arrêt des serveurs..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Capturer Ctrl+C pour nettoyer proprement
trap cleanup SIGINT

echo "Vérification des dépendances..."

# Vérifier Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 n'est pas installé"
    exit 1
fi

# Vérifier Node.js
if ! command -v node &> /dev/null; then
    echo "Node.js n'est pas installé"
    exit 1
fi

# Vérifier Angular CLI
if ! command -v ng &> /dev/null; then
    echo "Installation d'Angular CLI..."
    npm install -g @angular/cli
fi

echo "🔧 Installation des dépendances backend..."
cd backend
if [ ! -d "venv" ]; then
    echo "Création de l'environnement virtuel Python..."
    python3 -m venv venv
fi

source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null
pip install -r requirements.txt > /dev/null 2>&1

echo "Démarrage du serveur backend (port 8000)..."
uvicorn main:app --host 0.0.0.0 --port 8000 > ../backend.log 2>&1 &
BACKEND_PID=$!

cd ../frontend

echo "Installation des dépendances frontend..."
npm install > /dev/null 2>&1

echo "Démarrage du serveur frontend (port 4200)..."
ng serve --host 0.0.0.0 --port 4200 > ../frontend.log 2>&1 &
FRONTEND_PID=$!

echo "Attente du démarrage des serveurs..."
sleep 10

# Vérifier que les serveurs sont démarrés
if curl -s http://localhost:8000/ > /dev/null; then
    echo "Backend démarré avec succès sur http://localhost:8000"
else
    echo "Erreur lors du démarrage du backend"
    cat ../backend.log
    exit 1
fi

if curl -s http://localhost:4200/ > /dev/null; then
    echo "Frontend démarré avec succès sur http://localhost:4200"
else
    echo "Frontend en cours de démarrage..."
    sleep 15
    if curl -s http://localhost:4200/ > /dev/null; then
        echo "Frontend démarré avec succès sur http://localhost:4200"
    else
        echo "Erreur lors du démarrage du frontend"
        cat ../frontend.log
        exit 1
    fi
fi

echo ""
echo "Application démarrée avec succès !"
echo "Frontend: http://localhost:4200"
echo "Backend API: http://localhost:8000"
echo "Documentation API: http://localhost:8000/docs"
echo ""
echo "Appuyez sur Ctrl+C pour arrêter les serveurs"

# Attendre indéfiniment
wait

