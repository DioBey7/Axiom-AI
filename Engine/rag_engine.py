import os
import time
import base64
import io
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredPowerPointLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun

class RAGEngine:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        self.api_key = api_key.strip() if api_key else None
        self.vector_db = None
        self.retriever = None
        self.current_room = None
        
        if self.api_key:
            os.environ["GOOGLE_API_KEY"] = self.api_key
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=self.api_key
            )
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", 
                temperature=0.3,
                google_api_key=self.api_key
            )

    def set_room(self, room_name):
        self.current_room = room_name
        db_path = f"./workspaces/{room_name}/db"
        if not os.path.exists(db_path):
            os.makedirs(db_path)
        
        self.vector_db = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )
        self.retriever = self.vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 30}
        )

    def process_documents(self, uploaded_files, max_chunks):
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY eksik.")
        if not self.current_room:
            raise ValueError("Lütfen önce bir çalışma odası seçin.")
            
        all_docs = []
        if not os.path.exists("temp"):
            os.makedirs("temp")

        for uploaded_file in uploaded_files:
            file_path = os.path.join("temp", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                if file_path.lower().endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    loaded_docs = loader.load()
                elif file_path.lower().endswith(".txt"):
                    loader = TextLoader(file_path, encoding="utf-8")
                    loaded_docs = loader.load()
                elif file_path.lower().endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                    loaded_docs = loader.load()
                elif file_path.lower().endswith(".pptx"):
                    loader = UnstructuredPowerPointLoader(file_path)
                    loaded_docs = loader.load()
                elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = Image.open(file_path)
                    buffered = io.BytesIO()
                    img_format = img.format if img.format else "PNG"
                    img.save(buffered, format=img_format)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    vision_msg = HumanMessage(
                        content=[
                            {"type": "text", "text": "Sen bir uzman sistem mimarısın. Bu görseldeki tüm teknik verileri, yapıları, ilişkileri ve metinleri eksiksiz, detaylı ve yapılandırılmış bir analitik metne dönüştür."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                        ]
                    )
                    vision_response = self.llm.invoke([vision_msg])
                    doc = Document(
                        page_content=f"[GÖRSEL ANALİZİ - {uploaded_file.name}]:\n{vision_response.content}",
                        metadata={"source": uploaded_file.name, "page": 1}
                    )
                    loaded_docs = [doc]
                else:
                    st.warning(f"{uploaded_file.name} desteklenmeyen format.")
                    continue
                
                all_docs.extend(loaded_docs)
            except Exception as e:
                st.error(f"{uploaded_file.name} işleme hatası: {str(e)}")

        if not all_docs:
            raise ValueError("İndekslenecek geçerli bir veri bulunamadı.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, 
            chunk_overlap=300,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(all_docs)
        total_chunks = len(chunks)
        safe_chunks = chunks[:max_chunks]

        # Vektörleştirme aşamasına hata direnci (Retry Logic) eklendi
        db_path = f"./workspaces/{self.current_room}/db"
        for attempt in range(3):
            try:
                self.vector_db = Chroma.from_documents(
                    documents=safe_chunks, 
                    embedding=self.embeddings,
                    persist_directory=db_path
                )
                break
            except Exception as e:
                if attempt < 2 and any(err in str(e) for err in ["500", "503", "429"]):
                    time.sleep(5 * (attempt + 1))
                    continue
                raise e

        self.retriever = self.vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 30}
        )
        
        return total_chunks, len(safe_chunks)

    def get_response(self, query, raw_history, profession, use_web):
        if not self.retriever:
            return "İşlem reddedildi. Lütfen önce veri kaynağı sağlayın.", []

        for attempt in range(3):
            try:
                formatted_history = []
                recent_history = raw_history[-4:] if len(raw_history) > 4 else raw_history
                
                for msg in recent_history:
                    if msg["role"] == "user":
                        formatted_history.append(HumanMessage(content=msg["content"]))
                    else:
                        formatted_history.append(AIMessage(content=msg["content"]))

                search_query = query
                if formatted_history:
                    contextualize_q_system_prompt = (
                        "Geçmiş sohbet ve son soru verildi. "
                        "Bunu veritabanında aramak için tek, kısa ve net bir arama sorgusuna çevir. "
                        "SADECE arama cümlesini yaz."
                    )
                    contextualize_prompt = ChatPromptTemplate.from_messages([
                        ("system", contextualize_q_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}")
                    ])
                    query_generator = contextualize_prompt | self.llm | StrOutputParser()
                    generated_query = query_generator.invoke({"input": query, "chat_history": formatted_history})
                    if generated_query and generated_query.strip() and len(generated_query) <= 200:
                        search_query = generated_query

                sources = self.retriever.invoke(search_query)
                context_text = "\n\n".join(doc.page_content for doc in sources)

                web_context = ""
                if use_web:
                    try:
                        search_tool = DuckDuckGoSearchRun()
                        web_context = search_tool.invoke(search_query)
                    except Exception:
                        web_context = "Web arama modülü şu an yanıt vermiyor."

                qa_system_prompt = (
                    "Sen {profession} disiplininde dünya çapında tanınan, vizyoner bir ordinaryüs profesör ve baş mimarsın. "
                    "Karşındaki öğrenci, senin akademik ve sektörel mirasını devralacak, detaylara hakim, mükemmeliyetçi bir mühendis adayı. "
                    "Amacın sadece bilgiyi sunmak değil; çok boyutlu (görsel, yapısal, analitik) bir öğrenme deneyimi kurgulayarak onun bilişsel sınırlarını zorlamaktır.\n\n"
                    "### BİLİŞSEL EĞİTİM MİMARİSİ (KESİN KURALLAR):\n"
                    "1. Çok Modlu Anlatım: Metinleri asla uzun bloklar halinde yığma. Her konuyu hiyerarşik başlıklar (###) ve kalın vurgularla yapılandır.\n"
                    "2. 🎨 ZORUNLU GÖRSELLEŞTİRME (MERMAID.JS): Sistem mimarileri veya süreçler anlatırken KESİNLİKLE '```mermaid' kod bloğu kullanarak profesyonel diyagramlar çiz!\n"
                    "3. Endüstriyel Simülasyon: Anlattığın teorik bilginin {profession} endüstrisinde tam olarak nasıl hayat bulduğunu somut vaka analizleriyle detaylandır.\n"
                    "4. 🚀 VİZYONER UFUK (Kapanış): Yanıtını HER ZAMAN '🚀 İleri Öğrenme ve Endüstriyel Ufuk' başlığıyla bitir.\n\n"
                    "Yerel Bağlam:\n{context}\n\n"
                    "Web Bağlamı:\n{web_context}"
                )
                
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", qa_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])

                qa_chain = qa_prompt | self.llm | StrOutputParser()
                
                answer = qa_chain.invoke({
                    "context": context_text,
                    "web_context": web_context,
                    "profession": profession,
                    "chat_history": formatted_history,
                    "input": query
                })
                
                return answer, sources

            except Exception as e:
                error_str = str(e)
                if any(err in error_str for err in ["429", "RESOURCE_EXHAUSTED", "500", "INTERNAL", "503"]):
                    if attempt < 2:
                        time.sleep(8 * (attempt + 1)) # Üstel bekleme
                        continue
                return f"### ⚠️ Ordinaryüs Notu: Teknik Bağlantı Kesintisi\n\nGoogle AI servisleri şu an yoğunluk nedeniyle isteğimizi reddetti (`{error_str}`). Sistem 3 kez deneme yaptı ancak sonuç alamadı. Lütfen **1-2 dakika bekleyip** tekrar deneyin.", []