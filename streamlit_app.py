"""
TÜBİTAK 2204-A Lise Öğrencileri Araştırma Projeleri Asistanı
Streamlit + Google Gemini destekli proje metni üretim ve jüri değerlendirme aracı
"""

import html
import json
import re
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import streamlit as st

# ===================== SAYFA AYARLARI =====================

st.set_page_config(
    page_title="TÜBİTAK 2204-A Proje Asistanı",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

TR_TZ = ZoneInfo("Europe/Istanbul")

# ===================== ÖZEL CSS =====================

st.markdown(
    """
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #0f172a; margin-bottom: 0.3rem; }
    .sub-header { color: #64748b; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .section-box {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 1.2rem 1.4rem; margin-bottom: 1rem; white-space: pre-wrap;
    }
    .word-count { font-size: 0.85rem; color: #64748b; margin-top: 0.4rem; }
    .word-ok { color: #059669; font-weight: 600; }
    .word-warn { color: #d97706; font-weight: 600; }
    .word-bad { color: #dc2626; font-weight: 600; }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #0284c7, #0369a1);
        color: white; font-weight: 600; border: none;
        border-radius: 10px; padding: 0.7rem;
    }
    .info-box {
        background: #e0f2fe; border-left: 4px solid #0284c7;
        padding: 0.9rem 1.1rem; border-radius: 0 8px 8px 0;
        margin-bottom: 1.2rem; font-size: 0.95rem;
    }
    .warn-box {
        background: #fef3c7; border-left: 4px solid #d97706;
        padding: 0.7rem 1rem; border-radius: 0 8px 8px 0;
        margin-bottom: 1rem; font-size: 0.9rem; color: #78350f;
    }
    .gemini-badge {
        background: #d1fae5; color: #065f46; padding: 0.25rem 0.6rem;
        border-radius: 6px; font-size: 0.8rem; font-weight: 600;
    }
    .fallback-badge {
        background: #fef3c7; color: #92400e; padding: 0.25rem 0.6rem;
        border-radius: 6px; font-size: 0.8rem; font-weight: 600;
    }
    .jury-score-box {
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        border: 1px solid #bae6fd;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .jury-total {
        font-size: 2.4rem;
        font-weight: 800;
        color: #0369a1;
    }
    .jury-bar {
        height: 10px;
        border-radius: 5px;
        background: #e2e8f0;
        margin: 0.3rem 0 0.8rem 0;
        overflow: hidden;
    }
    .jury-bar-fill {
        height: 100%;
        border-radius: 5px;
        background: linear-gradient(90deg, #0284c7, #0ea5e9);
    }
    .jury-feedback {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-top: 0.8rem;
        font-size: 0.9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ===================== YARDIMCI FONKSİYONLAR =====================

def kelime_say(text: str) -> int:
    """Metindeki kelime sayısını hesaplar."""
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", text))


def kelime_durum(sayi: int, min_k: int, max_k: int) -> str:
    """Kelime sayısına göre HTML uyarı etiketi döndürür."""
    if min_k <= sayi <= max_k:
        return f'<span class="word-ok">✓ {sayi} kelime (uygun)</span>'
    if sayi < min_k:
        return f'<span class="word-warn">⚠ {sayi} kelime (en az {min_k} olmalı)</span>'
    return f'<span class="word-bad">✗ {sayi} kelime (en fazla {max_k} olmalı)</span>'


def temizle_fikir(text: str) -> str:
    """Boşlukları düzenler."""
    return re.sub(r"\s+", " ", text.strip())


def guvenli_html(text: str) -> str:
    """Kullanıcı / LLM çıktısını HTML olarak güvenle basar."""
    return html.escape(text or "")


# ===================== GEMINI MOTORU =====================

GEMINI_MODELS = ("gemini-2.0-flash", "gemini-1.5-flash")
REQUIRED_KEYS = ("proje_adi", "anahtar_kelimeler", "ozet", "amac", "yontem", "bulgular", "sonuc")


def _gemini_model(api_key: str):
    """Geçerli ilk modeli döndürür."""
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise RuntimeError(
            "`google-generativeai` paketi yüklü değil. `pip install google-generativeai` ile kurun."
        ) from e

    genai.configure(api_key=api_key)
    last_err: Optional[Exception] = None
    for name in GEMINI_MODELS:
        try:
            return genai.GenerativeModel(
                model_name=name,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_output_tokens": 4096,
                },
            )
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"Gemini modeli yüklenemedi: {last_err}")


def _extract_json(text: str) -> dict:
    """LLM çıktısından JSON bloğunu ayıklar."""
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    return json.loads(text.strip())


def gemini_uret(fikir: str, alan: str, yontem: str, seviye: str, api_key: str) -> dict:
    """2204-A formatında metin üretir."""
    model = _gemini_model(api_key)

    prompt = f"""Sen TÜBİTAK 2204-A Lise Öğrencileri Araştırma Projeleri Yarışması için resmi başvuru metni yazan bir uzmansın.

Görevin: Verilen ham proje fikrini, 2204-A resmi formatına uygun, akademik ve özgün bir iskelet metne dönüştürmek.

KURALLAR:
- Türkçe yaz.
- Proje adı en fazla 15 kelime olsun, net ve bilimsel olsun.
- Özet 150-250 kelime arasında olsun. Amaç, yöntem, temel bulgu ve sonucu kısaca içersin. Kaynak veya yorum ekleme.
- Amaç bölümü somut hedefler içersin.
- Yöntem bölümü seçilen yönteme uygun, adımlı ve bilimsel olsun.
- Bulgular bölümü "iskelet" olarak yazılsın; başında MUTLAKA şu not bulunsun: "(Not: Bu bölüm iskelet niteliğindedir; kendi ölçüm sonuçlarınızı, tablolarınızı ve grafiklerinizi eklemelisiniz.)"
- Sonuç bölümü araştırma sorusuyla ilişkilendirilsin, sınırlılık ve öneri içersin.
- Anahtar kelimeler tam 4 adet olsun.
- Hazır proje gibi durmasın, iskelet / taslak olduğu belli olsun.
- Marka, kurum adı, abartılı iddia kullanma.

GİRDİLER:
- Ham fikir: {fikir}
- Ana alan: {alan}
- Araştırma yöntemi: {yontem}
- Sınıf düzeyi: {seviye}

ÇIKTI FORMATI (sadece geçerli JSON döndür, başka hiçbir şey yazma):
{{
  "proje_adi": "...",
  "anahtar_kelimeler": "kelime1, kelime2, kelime3, kelime4",
  "ozet": "...",
  "amac": "...",
  "yontem": "...",
  "bulgular": "...",
  "sonuc": "..."
}}
"""

    response = model.generate_content(prompt)
    text = (getattr(response, "text", "") or "").strip()
    if not text:
        raise RuntimeError("Gemini boş yanıt döndürdü (güvenlik filtresi veya token limiti).")

    try:
        data = _extract_json(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Gemini JSON çıktısı ayrıştırılamadı: {e}") from e

    if not all(k in data for k in REQUIRED_KEYS):
        raise RuntimeError("Gemini yanıtında beklenen alanlardan bazıları eksik.")

    return {
        "adi": str(data["proje_adi"]).strip(),
        "anahtar": str(data["anahtar_kelimeler"]).strip(),
        "ozet": str(data["ozet"]).strip(),
        "amac": str(data["amac"]).strip(),
        "yontem": str(data["yontem"]).strip(),
        "bulgular": str(data["bulgular"]).strip(),
        "sonuc": str(data["sonuc"]).strip(),
        "kaynak": "gemini",
    }


# ===================== ŞABLON (FALLBACK) MOTORU =====================

STOPWORDS = {
    "için", "olan", "gibi", "daha", "ile", "bir", "bu", "ve", "veya", "olarak",
    "üzerine", "hakkında", "amaçlı", "yönelik", "ilgili", "çeşitli", "farklı",
    "konusunda", "olarak", "olduğu", "oldukça", "çok", "az", "en", "her",
}


def _baslik_uret(fikir: str, alan: str) -> str:
    kelimeler = fikir.split()
    baslik = " ".join(kelimeler[:10]).strip().capitalize() if kelimeler else "Araştırma Projesi"

    if alan and alan.lower() not in baslik.lower() and len(baslik.split()) < 10:
        baslik = f"{baslik}: {alan} Açısından Bir Araştırma"

    parcalar = baslik.split()
    if len(parcalar) > 15:
        baslik = " ".join(parcalar[:15])
    return baslik


def _anahtar_uret(fikir: str, alan: str) -> str:
    adaylar = [
        w.lower()
        for w in re.findall(r"\b[a-zA-ZçÇğĞıİöÖşŞüÜ]{4,}\b", fikir)
        if w.lower() not in STOPWORDS
    ]
    gorulen: set[str] = set()
    benzersiz = []
    for a in adaylar:
        if a not in gorulen:
            gorulen.add(a)
            benzersiz.append(a)

    anahtarlar = []
    if alan:
        anahtarlar.append(alan)
    anahtarlar.extend(benzersiz[: 4 - len(anahtarlar)])
    while len(anahtarlar) < 4:
        anahtarlar.append("lise araştırma projesi")
    return ", ".join(anahtarlar[:4]).title()


YONTEM_METINLERI = {
    "Deneysel (Kontrollü Deney)": (
        "Araştırma, kontrollü deney deseniyle yürütülmüştür. Bağımsız ve bağımlı değişkenler "
        "belirlenmiş, kontrol grubu ve deney grubu oluşturulmuştur.\n\n"
        "Veri toplama sürecinde ölçüm araçları kalibre edilmiş, tekrarlı ölçümler alınmış ve "
        "sonuçlar istatistiksel olarak karşılaştırılmıştır. Deney koşulları standardize edilerek "
        "dış etkenler kontrol altına alınmaya çalışılmıştır."
    ),
    "Tarama / Anket": (
        "Araştırma, tarama (survey) modeli kullanılarak gerçekleştirilmiştir. Hedef kitle "
        "belirlenmiş, örneklem seçilmiş ve veri toplama aracı olarak anket / ölçek uygulanmıştır.\n\n"
        "Anket maddeleri literatüre dayalı olarak hazırlanmış, uzman görüşü alınmış ve pilot "
        "uygulama yapılmıştır. Veriler betimsel ve çıkarımsal istatistik teknikleriyle analiz "
        "edilmiştir. Gerekli etik izinler ve onam formları alınmıştır."
    ),
    "Nitel (Görüşme / Gözlem)": (
        "Araştırma nitel desende yürütülmüştür. Veri toplama aracı olarak yarı yapılandırılmış "
        "görüşme formu ve/veya gözlem protokolü kullanılmıştır.\n\n"
        "Katılımcılardan elde edilen veriler içerik analizi / tematik analiz yöntemiyle "
        "çözümlenmiştir. Geçerlik ve güvenirlik için uzman görüşü, katılımcı teyidi ve veri "
        "çeşitlemesi tekniklerine başvurulmuştur."
    ),
    "Karma Yöntem": (
        "Araştırma karma yöntem deseniyle (nicel + nitel) yürütülmüştür. Nicel veriler anket / "
        "ölçek ile, nitel veriler ise görüşme veya açık uçlu sorularla toplanmıştır.\n\n"
        "İki veri seti birbirini destekleyecek şekilde analiz edilmiş ve bulgular bütünleştirilerek "
        "yorumlanmıştır."
    ),
    "Modelleme / Yazılım": (
        "Araştırma, model geliştirme / yazılım tabanlı bir yaklaşımla yürütülmüştür. Problem "
        "tanımlanmış, çözüm modeli tasarlanmış, prototip geliştirilmiş ve test edilmiştir.\n\n"
        "Performans metrikleri belirlenmiş, test sonuçları kaydedilmiş ve modelin başarısı "
        "değerlendirilmiştir."
    ),
}


def sablon_uret(fikir: str, alan: str, yontem: str, seviye: str) -> dict:
    fikir = temizle_fikir(fikir)
    baslik = _baslik_uret(fikir, alan)
    anahtar = _anahtar_uret(fikir, alan)

    ozet = f"""Bu araştırma, {fikir.lower()} konusunu {alan.lower()} alanı çerçevesinde ele almaktadır.
Lise düzeyinde yürütülen çalışma, mevcut literatürdeki boşlukları dikkate alarak özgün bir araştırma sorusu ortaya koymayı ve bilimsel yöntemlerle cevap üretmeyi amaçlamaktadır.

Araştırmanın temel amacı, belirlenen problemin sistematik bir şekilde incelenmesi, veri toplama araçlarıyla ölçülebilir bulgular elde edilmesi ve sonuçların bilimsel bir çerçevede yorumlanmasıdır. Bu doğrultuda {yontem.lower()} yöntemi tercih edilmiştir.

Çalışma kapsamında literatür taraması yapılmış, araştırma deseni oluşturulmuş, veri toplama süreci planlanmış ve elde edilen veriler uygun analiz teknikleriyle değerlendirilmiştir. Bulgular, projenin başlangıçta ortaya konulan araştırma sorusu ve hipotezleriyle ilişkilendirilerek tartışılmıştır.

Elde edilen sonuçların, {alan.lower()} alanına katkı sağlaması, benzer çalışmalara yol göstermesi ve lise düzeyinde bilimsel araştırma kültürünün gelişmesine destek olması beklenmektedir. Araştırma, öğrencilerin bilimsel düşünme, problem çözme ve veriye dayalı karar alma becerilerini geliştirmeyi de hedeflemektedir. Çalışmanın sınırlılıkları ve gelecekte yapılabilecek araştırmalar da sonuç bölümünde ele alınmıştır."""

    amac = f"""Bu projenin temel amacı, {fikir.lower()} konusunu bilimsel araştırma yöntemleriyle incelemek ve elde edilen bulgular ışığında {alan.lower()} alanına özgün katkılar sunmaktır.

Araştırmanın somut hedefleri şunlardır:
1. Konuya ilişkin mevcut literatürü tarayarak araştırma boşluklarını belirlemek.
2. Net ve ölçülebilir bir araştırma sorusu / hipotezi oluşturmak.
3. Uygun veri toplama araçlarını seçmek ve uygulamak.
4. Elde edilen verileri sistematik biçimde analiz etmek.
5. Bulguları bilimsel bir çerçevede yorumlayarak sonuç ve öneriler geliştirmek."""

    temel = YONTEM_METINLERI.get(yontem, YONTEM_METINLERI["Tarama / Anket"])
    yontem_metni = f"""{temel}

Araştırma süreci şu aşamalardan oluşmuştur:
• Literatür taraması ve problem tanımı
• Araştırma deseni ve örneklem belirleme
• Veri toplama araçlarının hazırlanması ve uygulanması
• Verilerin analizi
• Bulguların yorumlanması ve raporlaştırılması

Tüm süreçte bilimsel etik kurallara uyulmuş, gerekli izinler alınmıştır."""

    bulgular = """(Not: Bu bölüm iskelet niteliğindedir; kendi ölçüm sonuçlarınızı, tablolarınızı ve grafiklerinizi eklemelisiniz.)

Araştırma sonucunda elde edilen başlıca bulgular aşağıda özetlenmiştir:

• Veri analizi sonucunda araştırma sorusu / hipoteziyle ilişkili anlamlı sonuçlara ulaşılmıştır.
• Elde edilen sayısal / nitel bulgular, literatürdeki benzer çalışmalarla karşılaştırıldığında tutarlılık göstermiştir.
• Değişkenler arasındaki ilişkiler / farklar uygun analiz teknikleriyle değerlendirilmiştir.
• Beklenmeyen bulgular da tartışma bölümünde ele alınmıştır."""

    sonuc = f"""Bu araştırma, {fikir.lower()} konusunu {alan.lower()} perspektifinden inceleyerek alana katkı sunmayı hedeflemiştir.

Elde edilen bulgular ışığında şu sonuçlara varılmıştır:
• Araştırma sorusu cevaplanmış / hipotez test edilmiştir.
• Çalışma, lise düzeyinde bilimsel araştırma becerilerinin gelişmesine katkı sağlamıştır.
• Gelecekte yapılacak araştırmalar için öneriler ortaya konmuştur.

Araştırmanın sınırlılıkları arasında örneklem büyüklüğü, zaman kısıtı ve ölçüm araçlarının kapsamı sayılabilir. Bu sınırlılıklara rağmen çalışma, konuyla ilgili önemli veriler sunmakta ve benzer araştırmalara zemin hazırlamaktadır.

Öğrencilerin bilimsel düşünme, veri toplama, analiz etme ve rapor yazma becerileri bu süreçte gelişmiştir."""

    return {
        "adi": baslik,
        "anahtar": anahtar,
        "ozet": ozet.strip(),
        "amac": amac.strip(),
        "yontem": yontem_metni.strip(),
        "bulgular": bulgular.strip(),
        "sonuc": sonuc.strip(),
        "kaynak": "sablon",
    }


# ===================== JÜRİ SİMÜLASYONU =====================

def _heuristic_score(text: str, keywords: list, min_len: int = 80) -> float:
    """Basit uzunluk + anahtar kelime tabanlı 0-1 skor."""
    if not text:
        return 0.3
    t = text.lower()
    length_score = min(1.0, len(text) / max(min_len, 1))
    kw_hits = sum(1 for k in keywords if k.lower() in t)
    kw_score = min(1.0, kw_hits / max(len(keywords), 1) * 1.5)
    return round(0.45 * length_score + 0.55 * kw_score, 3)


def sablon_juri(proje: dict) -> dict:
    """Şablon motoru ile 0-100 jüri puanı (heuristik)."""
    problem_kw = ["problem", "araştırma sorusu", "hipotez", "amaç", "hedef", "ihtiyaç"]
    yontem_kw = ["yöntem", "deseni", "veri", "örneklem", "analiz", "ölçüm", "anket", "deney", "gözlem"]
    yarat_kw = ["özgün", "yenilik", "yenilikçi", "katkı", "etki", "potansiyel", "çözüm"]
    sonuc_kw = ["bulgu", "sonuç", "tartışma", "sınırlılık", "öneri", "tekrarlanabilir"]

    p1 = _heuristic_score(proje.get("amac", "") + " " + proje.get("ozet", ""), problem_kw, 120)
    p2 = _heuristic_score(proje.get("yontem", ""), yontem_kw, 150)
    p3 = _heuristic_score(proje.get("ozet", "") + " " + proje.get("sonuc", ""), yarat_kw, 100)
    p4 = _heuristic_score(proje.get("bulgular", "") + " " + proje.get("sonuc", ""), sonuc_kw, 120)
    p4 = max(0.35, p4 * 0.85)

    skorlar = {
        "problem": round(p1 * 20, 1),
        "yontem": round(p2 * 20, 1),
        "yaraticilik": round(p3 * 30, 1),
        "sonuclar": round(p4 * 30, 1),
    }
    toplam = round(sum(skorlar.values()), 1)

    yorumlar = []
    if skorlar["problem"] < 12:
        yorumlar.append("Problem tanımı ve araştırma sorusu daha net ve ölçülebilir hedeflerle güçlendirilmeli.")
    if skorlar["yontem"] < 12:
        yorumlar.append("Yöntem bölümü adım adım, veri toplama araçları ve analiz tekniği belirtilerek detaylandırılmalı.")
    if skorlar["yaraticilik"] < 18:
        yorumlar.append("Özgünlük ve potansiyel etki (toplumsal/çevresel/teknolojik) daha belirgin vurgulanmalı.")
    if skorlar["sonuclar"] < 18:
        yorumlar.append("Bulgular bölümüne gerçek veriler, tablolar ve tekrarlanabilirlik açıklaması eklenmeli (şu an iskelet).")
    if not yorumlar:
        yorumlar.append("İskelet metin genel olarak tutarlı. Gerçek veri ve özgün bulgularla zenginleştirildiğinde puan artacaktır.")

    return {
        "skorlar": skorlar,
        "toplam": toplam,
        "yorumlar": yorumlar,
        "kaynak": "sablon",
    }


def gemini_juri(proje: dict, api_key: str) -> dict:
    """Gemini ile resmi kriterlere göre jüri simülasyonu."""
    model = _gemini_model(api_key)

    prompt = f"""Sen TÜBİTAK 2204-A ön değerlendirme jürisisin. Aşağıdaki proje iskeletini resmi kriterlere göre puanla.

KRİTERLER (Önem Ağırlığı Puanı - ÖAP):
1. Problemin/ihtiyacın Tanımı, Araştırma Sorusu (max 20)
   - Ana alan uyumu, problem ve araştırma sorusunun açıklığı, hedeflerin ölçülebilirliği
2. Yöntem (max 20)
   - Yöntemin amaca uygunluğu, tekniklerin ayrıntılı ve açık ifade edilmesi
3. Yaratıcılık ve Etki (max 30)
   - Yenilikçi yaklaşım, alan/teknoloji/ekonomi/çevre/toplum üzerinde potansiyel etki
4. Sonuçlar ve Tartışma (max 30)
   - Sonuçların açıklık/doğruluk/tekrarlanabilirlik, kanıtların hedefleri desteklemesi

NOT: Bu bir İSKELET metindir; gerçek deney/anket verisi yoktur. Buna göre adil ama gerçekçi puan ver.
Öğrenci seviyesinde (lise) değerlendir.

PROJE:
Adı: {proje.get('adi','')}
Alan: {proje.get('alan','')}
Özet: {proje.get('ozet','')}
Amaç: {proje.get('amac','')}
Yöntem: {proje.get('yontem','')}
Bulgular: {proje.get('bulgular','')}
Sonuç: {proje.get('sonuc','')}

SADECE şu JSON formatında cevap ver, başka hiçbir şey yazma:
{{
  "problem": 0-20 arası sayı,
  "yontem": 0-20 arası sayı,
  "yaraticilik": 0-30 arası sayı,
  "sonuclar": 0-30 arası sayı,
  "yorumlar": ["kısa öneri 1", "kısa öneri 2", "kısa öneri 3"]
}}
"""

    response = model.generate_content(prompt)
    text = (getattr(response, "text", "") or "").strip()
    if not text:
        raise RuntimeError("Gemini jüri boş yanıt döndürdü.")

    data = _extract_json(text)

    skorlar = {
        "problem": float(data.get("problem", 10)),
        "yontem": float(data.get("yontem", 10)),
        "yaraticilik": float(data.get("yaraticilik", 15)),
        "sonuclar": float(data.get("sonuclar", 15)),
    }
    skorlar["problem"] = max(0.0, min(20.0, skorlar["problem"]))
    skorlar["yontem"] = max(0.0, min(20.0, skorlar["yontem"]))
    skorlar["yaraticilik"] = max(0.0, min(30.0, skorlar["yaraticilik"]))
    skorlar["sonuclar"] = max(0.0, min(30.0, skorlar["sonuclar"]))

    toplam = round(sum(skorlar.values()), 1)
    yorumlar = data.get("yorumlar") or ["Değerlendirme tamamlandı."]
    if isinstance(yorumlar, str):
        yorumlar = [yorumlar]

    return {
        "skorlar": {k: round(v, 1) for k, v in skorlar.items()},
        "toplam": toplam,
        "yorumlar": list(yorumlar)[:5],
        "kaynak": "gemini",
    }


def juri_yorum_renk(toplam: float) -> str:
    if toplam >= 75:
        return "#059669"
    if toplam >= 55:
        return "#d97706"
    return "#dc2626"


# ===================== ARAYÜZ =====================

st.markdown('<p class="main-header">🔬 TÜBİTAK 2204-A Proje Asistanı</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Lise Öğrencileri Araştırma Projeleri • Gemini destekli resmi formata uygun metin iskeleti</p>',
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="info-box">
<strong>Önemli:</strong> Bu araç yalnızca bir <strong>çıkış noktası / iskelet</strong> üretir.
Üretilen metinleri kendi özgün araştırmanıza, ölçüm sonuçlarınıza ve okul imkânlarınıza göre
mutlaka geliştirip zenginleştirmeniz gerekmektedir. Hazır proje sunmak yasaktır.
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Ayarlar")

    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""

    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        value=st.session_state["api_key"],
        help=(
            "https://aistudio.google.com/apikey adresinden ücretsiz alabilirsiniz. "
            "Boş bırakırsanız şablon motoru kullanılır."
        ),
        placeholder="AIza...",
    )
    st.session_state["api_key"] = api_key

    st.markdown("---")
    st.header("Proje Parametreleri")

    alan = st.selectbox(
        "Ana Alan",
        [
            "Biyoloji", "Kimya", "Fizik", "Matematik",
            "Bilgisayar / Yazılım", "Çevre Bilimleri", "Gıda / Tarım",
            "Sosyal Bilimler", "Psikoloji / Eğitim", "Mühendislik", "Diğer",
        ],
    )

    yontem = st.selectbox(
        "Araştırma Yöntemi",
        [
            "Deneysel (Kontrollü Deney)",
            "Tarama / Anket",
            "Nitel (Görüşme / Gözlem)",
            "Karma Yöntem",
            "Modelleme / Yazılım",
        ],
    )

    seviye = st.selectbox(
        "Sınıf Düzeyi",
        ["9. Sınıf", "10. Sınıf", "11. Sınıf", "12. Sınıf", "Hazırlık"],
    )

    st.markdown("---")
    if api_key:
        st.success("🟢 Gemini aktif")
    else:
        st.info("🟡 Şablon motoru (API key yok)")
    st.caption("2204-A Proje Rehberi Uyumlu")
    st.caption("Özet: 150-250 kelime • Proje adı: max 15 kelime")


# Ana girdi
st.subheader("1. Ham Proje Fikrinizi Yazın")
ham_fikir = st.text_area(
    "Proje fikrinizi detaylı yazın (ne araştırılacak, neden önemli, nasıl yapılacak...)",
    height=140,
    placeholder=(
        "Örnek: Okulumuzdaki öğrencilerin günlük su tüketim alışkanlıklarının incelenmesi "
        "ve su tasarrufu bilincinin artırılmasına yönelik bir farkındalık çalışması..."
    ),
)

uret_btn = st.button("🚀 Resmi 2204-A Formatına Dönüştür", type="primary")


# Üretim
if uret_btn:
    if not ham_fikir or len(ham_fikir.strip()) < 20:
        st.error("Lütfen en az 20 karakterlik anlamlı bir proje fikri girin.")
    else:
        with st.spinner("Proje metni oluşturuluyor..."):
            sonuc: Optional[dict] = None
            if api_key:
                try:
                    sonuc = gemini_uret(ham_fikir, alan, yontem, seviye, api_key)
                except Exception as e:  # noqa: BLE001
                    st.warning(
                        f"Gemini yanıtı alınamadı, şablon motoruna geçiliyor. "
                        f"({type(e).__name__}: {e})"
                    )

            if sonuc is None:
                sonuc = sablon_uret(ham_fikir, alan, yontem, seviye)

            st.session_state["proje"] = {
                **sonuc,
                "alan": alan,
                "yontem_secim": yontem,
                "seviye": seviye,
                "tarih": datetime.now(TR_TZ).strftime("%d.%m.%Y %H:%M"),
            }
            if "juri" in st.session_state:
                del st.session_state["juri"]


# Sonuçları göster
if "proje" in st.session_state:
    p = st.session_state["proje"]

    if p.get("alan") != alan or p.get("yontem_secim") != yontem or p.get("seviye") != seviye:
        st.markdown(
            '<div class="warn-box">⚠️ Parametreler değişti. Güncel çıktı için yeniden üretin.</div>',
            unsafe_allow_html=True,
        )

    if p.get("kaynak") == "gemini":
        st.markdown('<span class="gemini-badge">🟢 Gemini ile üretildi</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="fallback-badge">🟡 Şablon motoru ile üretildi</span>', unsafe_allow_html=True)

    st.success("Metinler oluşturuldu. Aşağıdaki bölümleri kopyalayıp kendi araştırmanıza göre düzenleyin.")

    # Proje Adı
    st.markdown("### 📌 Proje Adı (max 15 kelime)")
    st.code(p["adi"], language="text")
    st.markdown(
        f'<p class="word-count">{kelime_durum(kelime_say(p["adi"]), 3, 15)}</p>',
        unsafe_allow_html=True,
    )

    # Anahtar Kelimeler
    st.markdown("### 🔑 Anahtar Kelimeler (4 adet)")
    st.code(p["anahtar"], language="text")

    # Özet
    st.markdown("### 📝 Proje Özeti (150-250 kelime)")
    st.markdown(f'<div class="section-box">{guvenli_html(p["ozet"])}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="word-count">{kelime_durum(kelime_say(p["ozet"]), 150, 250)}</p>',
        unsafe_allow_html=True,
    )

    # Amaç
    st.markdown("### 🎯 Amaç")
    st.markdown(f'<div class="section-box">{guvenli_html(p["amac"])}</div>', unsafe_allow_html=True)

    # Yöntem
    st.markdown("### 🔬 Yöntem")
    st.markdown(f'<div class="section-box">{guvenli_html(p["yontem"])}</div>', unsafe_allow_html=True)

    # Bulgular
    st.markdown("### 📊 Bulgular (İskelet)")
    st.markdown(f'<div class="section-box">{guvenli_html(p["bulgular"])}</div>', unsafe_allow_html=True)
    st.info("⚠️ Bu bölümü kendi gerçek ölçüm / anket / deney sonuçlarınızla doldurmanız zorunludur.")

    # Sonuç
    st.markdown("### ✅ Sonuç ve Öneriler")
    st.markdown(f'<div class="section-box">{guvenli_html(p["sonuc"])}</div>', unsafe_allow_html=True)

    # İndirme
    st.markdown("---")
    st.subheader("📥 Metni İndir")

    motor = "Gemini" if p.get("kaynak") == "gemini" else "Şablon"
    tam_metin = f"""TÜBİTAK 2204-A Lise Öğrencileri Araştırma Projeleri Yarışması
Proje Metni İskeleti
Oluşturulma: {p['tarih']}
Alan: {p['alan']} | Yöntem: {p['yontem_secim']} | Sınıf: {p.get('seviye', '-')}
Motor: {motor}

================================================
PROJE ADI
================================================
{p['adi']}

================================================
ANAHTAR KELİMELER
================================================
{p['anahtar']}

================================================
PROJE ÖZETİ
================================================
{p['ozet']}

================================================
AMAÇ
================================================
{p['amac']}

================================================
YÖNTEM
================================================
{p['yontem']}

================================================
BULGULAR
================================================
{p['bulgular']}

================================================
SONUÇ VE ÖNERİLER
================================================
{p['sonuc']}

------------------------------------------------
UYARI: Bu metin yalnızca bir iskelettir.
Kendi özgün araştırmanız, verileriniz ve yorumlarınızla mutlaka zenginleştirilmelidir.
"""

    st.download_button(
        label="📄 Tüm Metni TXT olarak indir",
        data=tam_metin,
        file_name=f"2204A_Proje_Iskeleti_{datetime.now(TR_TZ).strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
    )

    st.caption("İpucu: İndirdiğiniz metni Word'e yapıştırıp resmi Proje Yazım Şablonuna göre düzenleyebilirsiniz.")

    # ---------- JÜRİ SİMÜLASYONU ----------
    st.markdown("---")
    st.subheader("⚖️ Resmi Jüri Simülasyonu (Ön Değerlendirme)")

    st.caption(
        "TÜBİTAK 2204-A Ön Değerlendirme kriterlerine göre "
        "(Problemin Tanımı 20 + Yöntem 20 + Yaratıcılık/Etki 30 + Sonuçlar 30 = 100). "
        "Bu bir simülasyondur, resmi sonuç değildir."
    )

    juri_btn = st.button("⚖️ Jüri Değerlendirmesini Başlat", key="juri_btn")

    if juri_btn:
        with st.spinner("Jüri değerlendiriyor..."):
            juri_sonuc = None
            api_key_now = st.session_state.get("api_key", "")
            if api_key_now:
                try:
                    juri_sonuc = gemini_juri(p, api_key_now)
                except Exception as e:  # noqa: BLE001
                    st.warning(
                        f"Gemini jüri başarısız, şablon jüriye geçiliyor. ({type(e).__name__})"
                    )
            if juri_sonuc is None:
                juri_sonuc = sablon_juri(p)
            st.session_state["juri"] = juri_sonuc

    if "juri" in st.session_state:
        j = st.session_state["juri"]
        skorlar = j["skorlar"]
        toplam = j["toplam"]
        renk = juri_yorum_renk(toplam)

        if j.get("kaynak") == "gemini":
            st.markdown(
                '<span class="gemini-badge">🟢 Gemini Jüri</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="fallback-badge">🟡 Şablon Jüri</span>',
                unsafe_allow_html=True,
            )

        if toplam >= 70:
            durum_yazi = "Güçlü iskelet – gerçek veriyle yükselme potansiyeli yüksek"
        elif toplam >= 50:
            durum_yazi = "Orta düzey – yöntem ve bulgular güçlendirilmeli"
        else:
            durum_yazi = "Zayıf iskelet – problem tanımı ve yöntem detaylandırılmalı"

        st.markdown(
            f'<div class="jury-score-box">'
            f'<div style="display:flex; align-items:baseline; gap:0.5rem;">'
            f'<span class="jury-total" style="color:{renk}">{toplam}</span>'
            f'<span style="font-size:1.1rem; color:#64748b;">/ 100</span></div>'
            f'<p style="margin:0.3rem 0 1rem 0; color:#475569; font-size:0.95rem;">{durum_yazi}</p>'
            f"</div>",
            unsafe_allow_html=True,
        )

        kriterler = [
            ("Problemin Tanımı / Araştırma Sorusu", "problem", 20),
            ("Yöntem", "yontem", 20),
            ("Yaratıcılık ve Etki", "yaraticilik", 30),
            ("Sonuçlar ve Tartışma", "sonuclar", 30),
        ]
        for ad, key, max_p in kriterler:
            val = skorlar.get(key, 0)
            pct = min(100.0, (val / max_p) * 100)
            st.markdown(f"**{ad}** — {val} / {max_p}")
            st.markdown(
                f'<div class="jury-bar"><div class="jury-bar-fill" style="width:{pct}%"></div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("#### 📋 Jüri Önerileri")
        for y in j.get("yorumlar", []):
            st.markdown(
                f'<div class="jury-feedback">• {guvenli_html(str(y))}</div>',
                unsafe_allow_html=True,
            )

        st.info(
            "💡 Bu simülasyon eğitim amaçlıdır. Gerçek jüri değerlendirmesi akademisyenler "
            "tarafından proje raporu, veriler ve sunum üzerinden yapılır. İskelet metin "
            "olduğu için özellikle **Sonuçlar** puanı düşük kalabilir; kendi verilerinizi "
            "ekledikçe artacaktır."
        )


# Footer
st.markdown("---")
st.caption(
    "Bu araç eğitim ve rehberlik amaçlıdır. Resmi başvuru için TÜBİTAK'ın güncel "
    "Proje Rehberi ve Yazım Şablonunu mutlaka inceleyiniz."
)