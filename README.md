# 🏛️ Axiom AI: Bilişsel Öğrenme ve Mimari Ekosistemi

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge)
![Google Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B6B?style=for-the-badge)

Axiom AI, salt bilgi veren standart sohbet asistanlarının aksine; **Sistem Mimarisi, Yazılım Mühendisliği ve Derin Analitik** süreçleri için özel olarak tasarlanmış, **Çok Modlu (Multimodal) RAG** destekli otonom bir mentörlük ekosistemidir.

Öğrenciye sadece kodu veya teoriyi vermekle kalmaz; arka plandaki "Neden?" sorusunu cevaplar, sektörel kullanım senaryoları (use-cases) sunar ve `Mermaid.js` tabanlı görsel akış şemalarıyla bilişsel hafızayı tetikler.

## 🛠️ Sistem Mimarisi ve Kullanılan Teknolojiler (Tech Stack)

Axiom AI, modern yapay zeka araçlarının ve veri işleme kütüphanelerinin birbirine entegre edildiği hibrit bir mimari üzerine kurulmuştur:

* 🧠 **Yapay Zeka Motoru (LLM & Vision):** `Google Gemini 2.5 Flash` *(Hem metin sentezi hem de otonom görsel analiz - Vision RAG - için kullanıldı)*
* ⚙️ **RAG Çerçevesi (Framework):** `LangChain` *(Core, Community ve Google GenAI modülleri ile vektör-model köprüsü kuruldu)*
* 🗄️ **Vektör Veritabanı:** `ChromaDB` *(Yerel, izole ve kalıcı 'Çalışma Odası' hafızaları için tercih edildi)*
* 🎨 **Arayüz (Frontend):** `Streamlit` *(Standart UI yerine özel CSS enjeksiyonu, Glassmorphism ve Dark Academia teması ile baştan tasarlandı)*
* 📄 **Veri Boru Hattı (Data Pipeline):** * `PyPDF` (PDF ayrıştırma)
  * `Docx2txt` (Word belgeleri)
  * `Unstructured` (Karmaşık PPTX sunumları)
  * `Pillow` (Görüntü/Piksel işleme)
* 🌐 **Otonom Ajan (Web Fallback):** `DuckDuckGo Search API` *(Yerel bağlamda bulunmayan ekstrem senaryolar için otonom internet araması)*
* 🖨️ **Dışa Aktarma (Export) Motoru:** `FPDF2` *(Kütüphanenin varsayılan render hatalarını aşmak için Python'un yerleşik `textwrap` modülüyle donanım seviyesinde güçlendirildi)*

## 🚀 Temel Mühendislik Özellikleri

* **👁️ Çok Modlu Görme Yeteneği (Vision RAG):** PDF, DOCX, TXT ve PPTX gibi klasik dökümanların yanı sıra, sisteme yüklenen mimari diyagramları, UML şemalarını ve algoritmik çizimleri (`PNG`, `JPG`) otonom olarak analiz eder ve vektör uzayına entegre eder.
* **🛡️ Dirençli Mimari (Self-Healing API):** Ağ trafiği veya Google Gemini API kotalarının aşılması (429 RESOURCE_EXHAUSTED) durumunda sistem çökmez. Arka planda `Exponential Backoff` (kademeli geri çekilme) stratejisi uygulayarak bağlantıyı kendi kendine onarır ve işlemi sessizce yeniler.
* **🧠 Sokratik ve Proaktif Mentörlük:** Klasik RAG sistemlerinin aksine pedagojik bir yaklaşım kullanır. Her yanıtın sonunda "🚀 İleri Öğrenme ve Mesleki Ufuk" modülünü tetikleyerek kullanıcıya öğrenmesi gereken yeni nesil framework'leri ve algoritmaları önerir.
* **📂 Otonom Çalışma Odaları (Isolated Workspaces):** Her bir ders veya proje için izole edilmiş `ChromaDB` vektör veritabanları ve kalıcı sohbet hafızaları (`JSON`) oluşturur. Bağlam taşmasını (Context Overflow) önlemek için optimize edilmiş kayan pencere (Sliding Window) belleği kullanır.
* **🖨️ Kusursuz PDF İhracı (Custom Wrapping Engine):** `FPDF` kütüphanesinin yatay taşma (horizontal space) zafiyetlerini aşmak için özel bir metin parçalama (`textwrap`) algoritması kullanır. Üretilen derin analizleri sıfır hata toleransıyla PDF olarak dışa aktarır.
* **🌑 Minimalist Dark Academia Arayüz:** Göz yormayan monokrom renk paleti (Zinc/Slate), cam efekti (Glassmorphism) ve pürüzsüz CSS animasyonlarıyla kod okuma/yazma seansları için özel dizayn edilmiş premium kullanıcı arayüzü.

## ⚙️ Sistem Kurulumu

Projeyi yerel makinenizde (Localhost) çalıştırmak için aşağıdaki adımları sırasıyla uygulayın:

### 1. Repoyu Klonlayın
```bash
git clone [https://github.com/DioBey7/Axiom-AI.git](https://github.com/DioBey7/Axiom-AI.git)
cd Axiom-AI
```

### 2. Sanal Ortam (Virtual Environment) Oluşturun
Sistem bağımlılıklarının çakışmaması için izole bir ortam kurun:
```bash
python -m venv venv
```
Sanal ortamı aktif edin:
* Windows için: venv\Scripts\activate
* Mac/Linux için: source venv/bin/activate

### 3. Bağımlılıkları Yükleyin
Gerekli tüm kütüphaneleri ve AI araçlarını tek komutla kurun:
```bash
pip install -r requirements.txt
```

### 4. Çevre Değişkenlerini Ayarlayın
Proje ana dizininde (app.py ile aynı yerde) bir .env dosyası oluşturun ve Google Gemini API anahtarınızı içine ekleyin:
```bash
GOOGLE_API_KEY="AIzaSy_SİZİN_GİZLİ_API_ANAHTARINIZ_BURAYA"
```

### 5. Motoru Ateşleyin
Kurulum tamamlandı! Axiom AI'yi başlatmak için terminale şu komutu girin:
```bash
streamlit run app.py
```

## 🛠️ Kullanılan Teknolojiler (Tech Stack)
* LLM & Vision Engine: Google Gemini 2.5 Flash
* RAG Framework: LangChain (Core, Community, Google GenAI)
* Vektör Veritabanı: ChromaDB
* Arayüz (Frontend): Streamlit (Custom CSS & Glassmorphism)
* Döküman İşleyiciler: PyPDF, Unstructured (PPTX), Docx2txt, Pillow (Images)
* Dışa Aktarma Motoru: FPDF2 & Custom TextWrapper
* Ajan Yeteneği (Web Fallback): DuckDuckGo Search API

## 📬 Geliştirici Notu ve İletişim
Bu projeyi incelerken umarım siz de bir sistemin sınırlarını zorlamanın, sıfırdan otonom bir ekosistem inşa etmenin ve o terminal ekranındaki hata loglarını tek tek çözmenin verdiği mühendislik heyecanını hissedersiniz.

Mimari tasarımlara, veri yapılarına ve sistem optimizasyonlarına tutkulu bir yazılım mühendisliği öğrencisi olarak; açık kaynak dünyasındaki her türlü geri bildirime, fikir alışverişine ve ortak beyin fırtınalarına kapım sonuna kadar açık. Birlikte her zaman daha iyisini, daha güvenlisini ve daha performanslısını inşa edebiliriz!

Ağımı genişletmekten ve yeni teknolojiler üzerine konuşmaktan büyük keyif alıyorum. Bana buralardan ulaşabilirsiniz:
* 📧 E-Posta: [beyza04yazici2005@gmail.com]
* 🔗 LinkedIn: [www.linkedin.com/in/beyza-yazıcı-400183332]
* 🐙 GitHub: [https://www.google.com/search?q=https://github.com/DioBey7]

## 📄 Lisans
Bu proje **MIT Lisansı** altında lisanslanmıştır. Kodları incelemekte, kendi projelerinizde kullanmakta, değiştirmekte ve dağıtmakta tamamen özgürsünüz. Bilginin ve mühendisliğin paylaştıkça büyüdüğüne inanıyorum. Daha fazla detay için `LICENSE` dosyasına göz atabilirsiniz.
