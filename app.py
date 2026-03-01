import os
import json
import shutil
import re
import textwrap
import time
import base64
import io
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    Docx2txtLoader, 
    UnstructuredPowerPointLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from fpdf import FPDF

load_dotenv()

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
                model="gemini-embedding-001",
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

        db_path = f"./workspaces/{self.current_room}/db"
        self.vector_db = Chroma.from_documents(
            documents=safe_chunks, 
            embedding=self.embeddings,
            persist_directory=db_path
        )
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
                    "1. Çok Modlu Anlatım (Multimodal Pedagogy): Metinleri asla uzun bloklar halinde yığma. Her konuyu hiyerarşik başlıklar (###), kısa/vurucu paragraflar ve kalın/italik vurgularla yapılandır. Hap gibi ve estetik olsun.\n"
                    "2. 🎨 ZORUNLU GÖRSELLEŞTİRME (MERMAID.JS): Sistem mimarileri, veri yapıları, algoritmalar veya süreçler anlatırken KESİNLİKLE '```mermaid' kod bloğu kullanarak profesyonel akış şemaları, UML diyagramları veya durum makineleri çiz! Öğrencinin görsel zekasını ateşle.\n"
                    "3. Endüstriyel Simülasyon: Anlattığın teorik bilginin {profession} endüstrisinde (gerçek dünya projelerinde, devasa mimarilerde, AR-GE ortamlarında) tam olarak nasıl hayat bulduğunu somut vaka analizleriyle (use-cases) detaylandır.\n"
                    "4. Sokratik ve Analojik Derinlik: Sadece 'Nasıl' çalıştığını değil, 'Neden' o şekilde tasarlandığını öğret. Karmaşık yapıları günlük hayattan veya donanım seviyesinden vurucu analojilerle basitleştir.\n"
                    "5. Dürüstlük ve Veri Kaynağı: Yerel bağlamdan cevaplıyorsan hissettir. Bilgi dökümanda yoksa akademik bir dürüstlükle 'Bu kaynaklarda yer almıyor ancak güncel literatüre/ağ verilerine göre...' şeklinde belirt.\n"
                    "6. 🚀 VİZYONER UFUK (Kapanış): Yanıtını HER ZAMAN '🚀 İleri Öğrenme ve Endüstriyel Ufuk' başlığıyla bitir. Öğrenciyi anlattığın konunun bir adım ötesine taşıyacak, modern endüstride onu öne çıkaracak en az 2 yenilikçi framework, teknoloji veya mimari strateji öner.\n\n"
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
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < 2:
                        time.sleep(8)
                        continue
                    else:
                        return "### ⏳ Sistem Uyarısı: Ağ Trafiği Sınırı\n\nGoogle API sınırlarına ulaştık. Sistem arka planda kendini onarmaya çalıştı ancak limitler şu an kapalı. Lütfen **1-2 dakika bekledikten sonra** sorunuzu tekrar iletin.", []
                return f"### ⚠️ Sistem İstisnası\nYanıt üretimi sırasında kritik bir hata meydana geldi: `{error_str}`", []

def get_workspaces():
    if not os.path.exists("./workspaces"):
        os.makedirs("./workspaces")
    dirs = [d for d in os.listdir("./workspaces") if os.path.isdir(os.path.join("./workspaces", d))]
    return dirs if dirs else ["Genel"]

def save_chat_history(room_name, messages):
    room_path = f"./workspaces/{room_name}"
    if not os.path.exists(room_path):
        os.makedirs(room_path)
    with open(f"{room_path}/history.json", "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)

def load_chat_history(room_name):
    history_path = f"./workspaces/{room_name}/history.json"
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def safe_clean_for_pdf(text):
    tr_map = {'ı':'i', 'İ':'I', 'ş':'s', 'Ş':'S', 'ğ':'g', 'Ğ':'G', 'ç':'c', 'Ç':'C', 'ö':'o', 'Ö':'O', 'ü':'u', 'Ü':'U'}
    for tr_char, eng_char in tr_map.items():
        text = text.replace(tr_char, eng_char)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.replace('**', '').replace('*', '').replace('#', '')
    return text

def generate_pdf_export(messages, room_name):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    title = safe_clean_for_pdf(f"{room_name} - Axiom Notlari")
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, text=title, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    for m in messages:
        role = "Ogrenci:" if m["role"] == "user" else "Profesor:"
        raw_content = safe_clean_for_pdf(m["content"])
        
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 8, text=role, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 10)
        
        lines = []
        for paragraph in raw_content.split('\n'):
            if not paragraph.strip():
                lines.append("")
            else:
                wrapped_lines = textwrap.wrap(paragraph, width=80, break_long_words=True, break_on_hyphens=False)
                lines.extend(wrapped_lines)
                
        for line in lines:
            pdf.cell(0, 6, text=line, new_x="LMARGIN", new_y="NEXT")
            
        pdf.ln(5)
        
    return bytes(pdf.output())

st.set_page_config(page_title="Axiom AI", page_icon="🏛️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    @import url('[https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap](https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap)');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp { background-color: #09090b; color: #e4e4e7; }
    
    div[data-testid="stSidebar"] { 
        background-color: #121214; 
        border-right: 1px solid #27272a;
    }
    
    h1, h2, h3, h4, h5, h6 { 
        color: #fafafa !important; 
        font-weight: 500 !important; 
        letter-spacing: -0.02em; 
    }
    
    h1 { font-size: 2rem; border-bottom: 1px solid #27272a; padding-bottom: 1rem; margin-bottom: 1.5rem; }
    
    .stTextInput input { 
        background-color: #09090b !important; 
        color: #fafafa !important; 
        border: 1px solid #27272a !important; 
        border-radius: 6px; 
        padding: 10px 14px; 
        font-size: 14px; 
        transition: border-color 0.2s ease;
    }
    .stTextInput input:focus { 
        border-color: #52525b !important; 
        box-shadow: none !important; 
    }
    
    .stChatMessage { 
        border-radius: 8px; 
        padding: 1.5rem; 
        margin-bottom: 1rem; 
        background-color: #09090b; 
        border: 1px solid #27272a; 
        line-height: 1.6; 
        font-size: 14px; 
        color: #d4d4d8;
    }
    
    .stButton>button { 
        background-color: #18181b; 
        color: #fafafa; 
        border: 1px solid #27272a; 
        border-radius: 6px; 
        font-weight: 500; 
        font-size: 14px;
        padding: 8px 16px; 
        transition: background-color 0.2s ease, border-color 0.2s ease; 
        min-height: 42px;
    }
    .stButton>button:hover { 
        background-color: #27272a; 
        border-color: #3f3f46; 
        color: #ffffff;
    }
    
    .stSelectbox div[data-baseweb="select"] > div { 
        background-color: #09090b; 
        color: #fafafa; 
        border-color: #27272a; 
        border-radius: 6px; 
    }
    
    hr { border-color: #27272a; margin: 1.5rem 0; }
    
    .source-card { 
        background-color: #09090b; 
        border-left: 3px solid #52525b; 
        padding: 16px; 
        margin-bottom: 8px; 
        border-radius: 4px; 
        border: 1px solid #27272a;
    }
    .source-title { color: #fafafa; font-weight: 500; margin-bottom: 4px; font-size: 14px; }
    .source-text { color: #a1a1aa; line-height: 1.5; font-size: 13px; }
    
    div[data-testid="stStatusWidget"] {
        background-color: #121214 !important;
        border: 1px solid #27272a !important;
        border-radius: 6px !important;
        color: #d4d4d8 !important;
    }
    </style>
    """, unsafe_allow_html=True)

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()

with st.sidebar:
    st.markdown("### 🗂️ Akademi Odaları")
    
    workspaces = get_workspaces()
    selected_room = st.selectbox("Mevcut Odalar", workspaces, key="room_selector", label_visibility="collapsed")
    
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    new_room_name = st.text_input("Yeni Oda Oluştur", placeholder="Örn: Algoritma_Analizi", label_visibility="collapsed")
    if st.button("Oda Ekle", use_container_width=True) and new_room_name:
        new_room_path = f"./workspaces/{new_room_name}"
        if not os.path.exists(new_room_path):
            os.makedirs(new_room_path)
            st.success(f"Oluşturuldu.")
            st.rerun()
        else:
            st.warning("Mevcut.")

    if "current_room" not in st.session_state or st.session_state.current_room != selected_room:
        st.session_state.current_room = selected_room
        st.session_state.rag_engine.set_room(selected_room)
        st.session_state.messages = load_chat_history(selected_room)
        st.session_state.total_chunks = 0
        st.session_state.indexed_chunks = 0
        st.session_state.loaded_files = []

    st.divider()
    st.markdown("### 🎯 Öğrenci Profili")
    profession = st.text_input("Uzmanlık Alanı", value="Yazılım Mühendisliği Öğrencisi")
    use_web = st.toggle("🌐 Akademik Ağ (Web Arama)", value=False, help="Eksik bilgiler için internette arama yapar.")

    st.divider()
    st.markdown("### 📥 Kütüphane Beslemesi")
    uploaded_files = st.file_uploader(
        f"Hedef: {st.session_state.current_room}", 
        type=['pdf', 'txt', 'docx', 'pptx', 'png', 'jpg', 'jpeg'], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    max_chunks = st.slider("İşlem Limiti", min_value=10, max_value=200, value=80)
    
    if uploaded_files:
        if st.button("Odaya İndeksle", use_container_width=True):
            with st.spinner("Materyaller işleniyor..."):
                try:
                    t_chunks, i_chunks = st.session_state.rag_engine.process_documents(uploaded_files, max_chunks)
                    st.session_state.total_chunks = t_chunks
                    st.session_state.indexed_chunks = i_chunks
                    st.session_state.loaded_files = [f.name for f in uploaded_files]
                    st.success("Entegrasyon başarılı.")
                except Exception as e:
                    st.error(f"Sistem hatası: {str(e)}")
    
    st.divider()
    st.markdown("### 💾 Veri Yönetimi")
    
    if st.session_state.messages:
        try:
            pdf_bytes = generate_pdf_export(st.session_state.messages, st.session_state.current_room)
            st.download_button(
                label="📄 Notları PDF İndir",
                data=pdf_bytes,
                file_name=f"{st.session_state.current_room}_axiom_notlari.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"PDF oluşturulamadı: {str(e)}")
    
    if st.button("Belleği Temizle", use_container_width=True):
        st.session_state.messages = []
        save_chat_history(st.session_state.current_room, [])
        st.rerun()

st.markdown(f"<h1>🏛️ Axiom AI | {st.session_state.current_room}</h1>", unsafe_allow_html=True)
st.caption("Bilişsel Öğrenme Ekosistemi | Vision RAG | Minimalist UI")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ordinaryüs Profesöre sorunuzu iletin..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Profesör bilişsel analizi başlatıyor...", expanded=True) as status:
            st.write("Vektör uzayı taranıyor ve kavramsal şemalar kurgulanıyor...")
            raw_history = st.session_state.messages[:-1]
            answer, sources = st.session_state.rag_engine.get_response(prompt, raw_history, profession, use_web)
            status.update(label="Bilişsel Analiz Tamamlandı.", state="complete", expanded=False)
            
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        save_chat_history(st.session_state.current_room, st.session_state.messages)
        
        if sources:
            with st.expander("📚 Akademik Kaynaklar (Gözlemlenebilirlik)"):
                for idx, doc in enumerate(sources):
                    kaynak_isim = os.path.basename(doc.metadata.get('source', 'Bilinmiyor'))
                    sayfa = doc.metadata.get('page', 'Yok')
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-title">Kaynak {idx+1}: {kaynak_isim} (Sayfa: {sayfa})</div>
                        <div class="source-text">"{doc.page_content[:250]}..."</div>
                    </div>
                    """, unsafe_allow_html=True)