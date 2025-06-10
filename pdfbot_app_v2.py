import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

# -------------------------------
# GESTION DE LA CLÉ API OPENAI
# -------------------------------

# Charger les variables depuis un fichier .env (s'il existe)
load_dotenv()

# Essayer d'obtenir la clé depuis les variables d’environnement
openai_api_key = os.getenv("OPENAI_API_KEY")

# -------------------------------
# 🎨 CONFIGURATION DE L’INTERFACE
# -------------------------------

# Définir les paramètres de la page
st.set_page_config(
    page_title="PDFBot",  # Titre de l'onglet
    page_icon="🤖",  # Icône dans l'onglet
    layout="centered"  # Centrage de la page
)

# -------------------------------
# 🧠 TITRE ET DESCRIPTION
# -------------------------------

st.title("🤖 PDFBot – Votre chatBot intelligent")
st.markdown(
    """Je votre assistant intelligent pour répondre à vos questions à partir du document PDF que vous me fournissez!""")

# Si la clé est absente, demander à l’utilisateur de la saisir dans la sidebar
if not openai_api_key:
    st.sidebar.title("Configuration API")
    openai_api_key = st.sidebar.text_input("Entrez votre clé OpenAI :", type="password")

# Si aucune clé n'est fournie, afficher une erreur et stopper l'app
if not openai_api_key:
    st.error("Veuillez entrer une clé API OpenAI pour continuer.")
    st.stop()

# -------------------------------
# 📄 TÉLÉVERSEMENT DU FICHIER PDF
# -------------------------------

uploaded_files = st.file_uploader(
    "📎 Téléversez un ou plusieurs fichiers PDF en même temps",
    type=["pdf"],
    accept_multiple_files=True
)

# Si aucun fichier n'est encore envoyé, afficher un message d’attente
if not uploaded_files:
    st.info("Veuillez séléctionner un ou plusieurs fichiers PDF en même temps! .")

# -------------------------------
# EXTRACTION DU TEXTE DU PDF
# -------------------------------

# Fonction utilitaire pour lire le PDF et en extraire le texte
def extract_text_from_multiple_pdfs(uploaded_files):
    """
    Prend une liste de fichiers PDF et retourne le texte combiné de tous.
    """
    all_text = ""

    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"

    return all_text

# -------------------------------
# DÉCOUPAGE DU TEXTE EN CHUNKS
# -------------------------------
# Fonction utilitaire pour le découpage du texte en chunks
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """
    Découpe le texte en petits morceaux de taille contrôlée.

    - chunk_size : taille maximale d’un morceau (en caractères).
    - chunk_overlap : nombre de caractères qui se chevauchent entre deux morceaux consécutifs.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.create_documents([text])
    return chunks

# -------------------------------
# EMBEDDINGS + FAISS
# -------------------------------

def create_faiss_index(chunks, api_key):
    """
    Crée une base FAISS à partir des chunks de texte.
    Chaque chunk est transformé en vecteur via OpenAIEmbeddings.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# ---------------------------
# Traitement du fichier PDF
# ---------------------------

# Si au moins un fichier est téléversé, on procède au traitement
if uploaded_files:
    with st.spinner("Lecture et extraction du texte en cours..."):
        text = extract_text_from_multiple_pdfs(uploaded_files)

    if not text.strip():
        st.error("Le fichier PDF semble vide ou ne contient pas de texte lisible.")
        st.stop()

    with st.spinner("Découpage du texte est en cours..."):
        chunks = split_text_into_chunks(text)

    # Création de l’index vectoriel
    with st.spinner("Création des embeddings et indexation vectorielle..."):
        vectorstore = create_faiss_index(chunks, openai_api_key)
        st.success("Base vectorielle FAISS créée avec succès.")

    # -------------------------------
    # 💬 Saisie de la question
    # -------------------------------

    st.markdown("---")
    st.header("💡 Posez une question sur votre document")

    question = st.text_input("❓ Votre question :")

    # Traitement de la question et recherche de réponse
    if question:
        with st.spinner("🔍 Recherche de la réponse..."):
            # 1. Recherche des documents les plus pertinents dans FAISS
            docs = vectorstore.similarity_search(question)

            # 2. Chargement du modèle de langage et de la chaîne de QA
            llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
            chain = load_qa_chain(llm, chain_type="stuff")

            # 3. Génération de la réponse
            result = chain.run(input_documents=docs, question=question)

            # 4. Affichage
            st.success("Réponse générée :")
            st.write(result)

            with st.expander("🔍 Documents sélectionnés (chunks)"):
                for doc in docs:
                    st.write(doc.page_content[:500])
