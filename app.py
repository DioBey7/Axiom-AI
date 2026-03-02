import os
import streamlit as st
from Engine.rag_engine import RAGEngine
from Utils.helpers import (
    get_workspaces, 
    save_chat_history, 
    load_chat_history, 
    generate_pdf_export
)

st.set_page_config(page_title="Axiom AI", page_icon="🏛️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    @import url('[https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap](https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap)');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header {visibility: hidden;}
    .stApp { background-color: #09090b; color: #e4e4e7; }
    div[data-testid="stSidebar"] { background-color: #121214; border-right: 1px solid #27272a; }
    h1 { font-size: 2rem; border-bottom: 1px solid #27272a; padding-bottom: 1rem; margin-bottom: 1.5rem; }
    .stTextInput input { background-color: #09090b !important; color: #fafafa !important; border: 1px solid #27272a !important; border-radius: 6px; }
    .stChatMessage { border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; background-color: #09090b; border: 1px solid #27272a; }
    .source-card { background-color: #09090b; border-left: 3px solid #52525b; padding: 16px; margin-bottom: 8px; border: 1px solid #27272a; }
    .source-title { color: #fafafa; font-weight: 500; font-size: 14px; }
    .source-text { color: #a1a1aa; font-size: 13px; }
    </style>
    """, unsafe_allow_html=True)

if "rag_engine" not in st.session_state:
    try:
        st.session_state.rag_engine = RAGEngine()
    except Exception as e:
        st.error(f"Sistem başlatılamadı: {str(e)}")

with st.sidebar:
    st.markdown("### 🗂️ Akademi Odaları")
    workspaces = get_workspaces()
    selected_room = st.selectbox("Mevcut Odalar", workspaces, key="room_selector", label_visibility="collapsed")
    
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    new_room_name = st.text_input("Yeni Oda Oluştur", placeholder="Örn: Algoritma_Analizi", label_visibility="collapsed")
    if st.button("Oda Ekle", use_container_width=True) and new_room_name:
        new_room_path = f"./workspaces/{new_room_name}"
        if not os.path.exists(new_room_path):
            try:
                os.makedirs(new_room_path)
                st.success("Oluşturuldu.")
                st.rerun()
            except Exception as e:
                st.error(f"Oda oluşturma hatası: {str(e)}")
        else:
            st.warning("Bu oda zaten mevcut.")

    if "current_room" not in st.session_state or st.session_state.current_room != selected_room:
        st.session_state.current_room = selected_room
        st.session_state.rag_engine.set_room(selected_room)
        st.session_state.messages = load_chat_history(selected_room)
        st.rerun()

    st.divider()
    st.markdown("### 🎯 Öğrenci Profili")
    profession = st.text_input("Uzmanlık Alanı", value="Yazılım Mühendisliği Öğrencisi")
    use_web = st.toggle("🌐 Akademik Ağ (Web Arama)", value=False)

    st.divider()
    st.markdown("### 📥 Kütüphane Beslemesi")
    uploaded_files = st.file_uploader(f"Hedef: {st.session_state.current_room}", type=['pdf', 'txt', 'docx', 'pptx', 'png', 'jpg', 'jpeg'], accept_multiple_files=True, label_visibility="collapsed")
    max_chunks = st.slider("İşlem Limiti", 10, 200, 80)
    
    if uploaded_files and st.button("Odaya İndeksle", use_container_width=True):
        with st.status("📚 Materyaller vektör uzayına işleniyor...") as status:
            try:
                t_chunks, i_chunks = st.session_state.rag_engine.process_documents(uploaded_files, max_chunks)
                status.update(label="✅ Entegrasyon Başarılı", state="complete")
                st.success(f"Toplam {t_chunks} parçadan {i_chunks} tanesi başarıyla indekslendi.")
            except Exception as e:
                status.update(label="❌ İşlem Başarısız", state="error")
                st.error(f"Kritik Hata: {str(e)}")
    
    st.divider()
    st.markdown("### 💾 Veri Yönetimi")
    
    if st.session_state.get("messages") and len(st.session_state.messages) > 0:
        try:
            with st.spinner("Rapor hazırlanıyor..."):
                pdf_bytes = generate_pdf_export(st.session_state.messages, st.session_state.current_room)
                st.download_button(
                    label="📄 Notları PDF İndir",
                    data=pdf_bytes,
                    file_name=f"{st.session_state.current_room}_axiom_notlari.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="pdf_download_btn"
                )
        except Exception as e:
            st.error(f"PDF Hazırlama Hatası: {str(e)}")
    
    if st.button("Belleği Temizle", use_container_width=True):
        st.session_state.messages = []
        save_chat_history(st.session_state.current_room, [])
        st.rerun()

st.markdown(f"<h1>🏛️ Axiom AI | {st.session_state.current_room}</h1>", unsafe_allow_html=True)
st.caption("Bilişsel Öğrenme Ekosistemi | Vision RAG | Modüler UI")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ordinaryüs Profesöre sorunuzu iletin..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("🧠 Profesör bilişsel analizi başlatıyor...", expanded=True) as status:
            try:
                raw_history = st.session_state.messages[:-1]
                answer, sources = st.session_state.rag_engine.get_response(prompt, raw_history, profession, use_web)
                status.update(label="🎓 Analiz Tamamlandı.", state="complete", expanded=False)
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                save_chat_history(st.session_state.current_room, st.session_state.messages)
                
                if sources:
                    with st.expander("📚 Akademik Kaynaklar"):
                        for idx, doc in enumerate(sources):
                            k_isim = os.path.basename(doc.metadata.get('source', 'Bilinmiyor'))
                            sayfa = doc.metadata.get('page', 'Yok')
                            st.markdown(f"""<div class="source-card"><div class="source-title">Kaynak {idx+1}: {k_isim} (Sayfa: {sayfa})</div><div class="source-text">"{doc.page_content[:250]}..."</div></div>""", unsafe_allow_html=True)
            except Exception as e:
                status.update(label="🚨 Sistem İstisnası", state="error")
                st.error(f"Yanıt üretilirken bir sorun oluştu: {str(e)}")

    with st.sidebar:
       st.divider()
       st.markdown("### 📄 Raporlama Merkezi")
       if st.session_state.get("messages") and len(st.session_state.messages) > 0:
           try:
               valid_messages = [m for m in st.session_state.messages if "⚠️" not in m["content"]]
               if valid_messages:
                   pdf_bytes = generate_pdf_export(st.session_state.messages, st.session_state.current_room)
                   st.download_button(
                       label="🏛️ Tüm Sohbeti PDF İndir",
                       data=pdf_bytes,
                       file_name=f"{st.session_state.current_room}_axiom_notlari.pdf",
                       mime="application/pdf",
                       use_container_width=True,
                       key="dynamic_pdf_btn"
                )
           except Exception as e:
               st.error("Rapor oluşturulurken teknik bir aksaklık yaşandı.")