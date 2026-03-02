import os
import json
import re
import textwrap
from fpdf import FPDF
from datetime import datetime

def get_workspaces():
    base_path = "./workspaces"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    try:
        dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        return sorted(dirs) if dirs else ["Genel"]
    except OSError:
        return ["Genel"]

def save_chat_history(room_name, messages):
    room_path = f"./workspaces/{room_name}"
    os.makedirs(room_path, exist_ok=True)
    file_path = os.path.join(room_path, "history.json")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=4)
    except Exception:
        pass

def load_chat_history(room_name):
    history_path = f"./workspaces/{room_name}/history.json"
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []

def safe_clean_for_pdf(text):
    tr_map = {
        'ı':'i', 'İ':'I', 'ş':'s', 'Ş':'S', 'ğ':'g', 'Ğ':'G', 
        'ç':'c', 'Ç':'C', 'ö':'o', 'Ö':'O', 'ü':'u', 'Ü':'U',
        '’': "'", '“': '"', '”': '"', '–': '-', '—': '-'
    }
    for tr_char, eng_char in tr_map.items():
        text = text.replace(tr_char, eng_char)
    
    text = re.sub(r'```mermaid.*?```', '[Gorsel Diyagram Atlandi]', text, flags=re.DOTALL)
    
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    text = text.replace('**', '').replace('*', '').replace('#', '').strip()
    return text

def generate_pdf_export(messages, room_name):
    pdf = FPDF()
    pdf.set_compression(True)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(20, 20, 40)
    pdf.cell(0, 12, text=f"AXIOM AI - {room_name.upper()} RAPORU", align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.cell(0, 5, text=f"Analiz Tarihi: {current_date} | Ordinaryus Profesor Modu", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    
    for m in messages:
        if "⚠️" in m["content"] or "Sistem Istisnasi" in m["content"] or "API Baglanti Siniri" in m["content"]:
            continue
            
        is_user = m["role"] == "user"
        role_label = "OGRENCI:" if is_user else "ORDINARYUS PROFESOR:"
        content = safe_clean_for_pdf(m["content"])
        
        pdf.set_font("helvetica", "B", 11)
        if is_user:
            pdf.set_fill_color(245, 245, 245)
            pdf.set_text_color(50, 50, 50)
        else:
            pdf.set_fill_color(230, 235, 245)
            pdf.set_text_color(30, 40, 100)
            
        pdf.cell(0, 8, text=f" {role_label}", fill=True, new_x="LMARGIN", new_y="NEXT")
        
        pdf.ln(2)
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        
        paragraphs = content.split('\n')
        for p in paragraphs:
            if p.strip():
                wrapped_lines = textwrap.wrap(p, width=95)
                for line in wrapped_lines:
                    pdf.cell(0, 6, text=line, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(1)
        pdf.ln(4)
        
    return bytes(pdf.output())