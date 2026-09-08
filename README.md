# 🫀 CardioSound AI — Heart Sound CAD System

تطبيق Streamlit لتشخيص أمراض القلب الصمامية من تسجيلات صوت القلب (PCG)، باستخدام
نموذج Hybrid Xception-CNN + LSTM مدرّب على بيانات MFCC و CWT scalograms.

هذا الفولدر جاهز للرفع مباشرة على GitHub ثم النشر (Deploy) على
[Streamlit Community Cloud](https://share.streamlit.io) — من غير ما تحتاج تشغّل أي حاجة
من داخل Colab.

---

## 1. هيكل المشروع

```
heart_cad_app/
├── app.py                  # نقطة الدخول (streamlit run app.py)
├── config.py                # كل الثوابت: مسارات، أسماء الكلاسات، الألوان، بيانات الفريق
├── styles.py                 # CSS للتطبيق (وضع فاتح/داكن)
├── audio_utils.py             # استخراج MFCC و CWT scalogram من ملف .wav
├── model_utils.py              # تحميل الموديل + الاستدلال (inference)
├── report_utils.py              # توليد تقرير PDF للتشخيص
├── components.py                 # عناصر واجهة قابلة لإعادة الاستخدام
├── dataset_lookup.py               # (اختياري) مطابقة الملف المرفوع بالداتاسيت الأصلي
├── views/                            # صفحات التطبيق (Home, Dataset, Upload, Diagnosis...)
├── models/                            # فاضي عمداً - الموديل بينزّل تلقائياً من Google Drive
├── assets/
├── .streamlit/config.toml              # ثيم التطبيق
└── requirements.txt
```

> ملحوظة: النوتبوك الأصلي اللي بعتّه (`Last_CAD_website_nns.ipynb`) هو Colab runner
> كان بيكتب نفس الملفات دي بأمر `%%writefile` ثم يشغّل التطبيق عبر tunnel. الملفات هنا
> هي نفس الكود لكن مستخرج كملفات حقيقية جاهزة لـ GitHub، مش نوتبوك.

---

## 2. الموديل المدرَّب (231 MB) — مربوط بـ Google Drive بالفعل ✅

GitHub بيرفض أي ملف أكبر من 100 MB، فالموديل `final_model.keras` **ما بيتراپع للـ repo
خالص**. بدل كده، `model_utils.py` بينزّله تلقائياً من Google Drive أول مرة التطبيق يشتغل
(ويحتفظ بيه بعد كده)، باستخدام مكتبة `gdown`.

الـ File ID بتاع الموديل بتاعك متظبط بالفعل جوه `config.py`:

```python
GDRIVE_MODEL_FILE_ID = "12RrMFoxWqAZlU2c8dD7C5o9q7uTwrFLB"   # final_model.keras (242 MB)
```

فمفيش خطوة إضافية مطلوبة منك — بس تأكد إن صلاحية المشاركة على الملف لسه
**"Anyone with the link"** وقت الـ deploy (لو غيّرتها، الموديل مش هينزّل).

> لو حابب تخفي الـ ID من الكود العام (لو الـ repo هيبقى Public)، تقدر تشيله من
> `config.py` وتحطه بدل كده في **Streamlit Cloud → App settings → Secrets**:
> ```toml
> GDRIVE_MODEL_FILE_ID = "12RrMFoxWqAZlU2c8dD7C5o9q7uTwrFLB"
> ```
> الـ Secrets دايماً بتلغي القيمة اللي في الكود.

ملف `mfcc_standardization_stats.npz` (إحصائيات التطبيع اللي اتحسبت وقت التدريب) مربوط
هو كمان بالفعل، فدقة التنبؤ هتبقى مطابقة تماماً لما كان وقت التدريب:

```python
GDRIVE_STATS_FILE_ID = "1WTCergkTSKSPpacLLkqsm_Le6eds6b6b"
```

---

## 3. الداتاسيت الأصلي (CWT + MFCC) — اختياري وإيقافه افتراضياً ⚠️

بعتّلي كمان لينكات فولدرات:
- `CWT_Scalogram_new299` (صور الـ CWT scalograms)
- `MFCC_Features_npy` (ملفات MFCC الخام)

الفولدرات دي بتُستخدم في ميزة اختيارية في `dataset_lookup.py`: لو المستخدم رفع ملف
`.wav` هو نفسه موجود ضمن عينات التدريب، التطبيق يجيب الـ features الحقيقية المحسوبة
مسبقاً بدل ما يعيد حسابها من الصوت الخام (ودقة أعلى + يعرف التشخيص الحقيقي تلقائياً).

**ليه إيقافها افتراضياً؟** الفولدرات دي عادة بتحتوي آلاف الملفات (1000 عينة × صورة +
مصفوفة لكل واحدة)، وممكن يكون حجمها كبير جداً بحيث:
- تاخد وقت طويل جداً تتنزل أول مرة يفتح فيها حد التطبيق.
- تتخطى مساحة التخزين المسموحة مجاناً على Streamlit Community Cloud.

فالـ repo مظبوط إنه **يشتغل بدونها تماماً** (التطبيق بيحسب MFCC/CWT من الصوت المرفوع
مباشرة - نفس الطريقة القياسية). لو عايز تجرب تفعيلها:

1. لينكات الفولدرات متظبطة بالفعل في `config.py`:
   ```python
   GDRIVE_CWT_FOLDER_ID = "1HMpxSnnQzBP3DJaPLsLavDgPVbAQWPlb"
   GDRIVE_MFCC_FOLDER_ID = "1bGmHEbC6cKFjuUhC1uKiuUjKrU4qprHR"
   ```
2. فعّل الميزة عن طريق Secret واحد بس في Streamlit Cloud:
   ```toml
   GDRIVE_SYNC_DATASET = "true"
   ```
3. جرّب على تطبيق تجريبي الأول (مش على النسخة اللي الناس بتستخدمها) عشان تتأكد إن
   الحجم والوقت مناسبين قبل ما تفعّلها بشكل دائم.

---

## 4. رفع المشروع على GitHub

### الطريقة 1 — من موقع GitHub مباشرة (الأسهل)
1. اعمل repo جديد على [github.com/new](https://github.com/new).
2. من صفحة الـ repo، دوس **"uploading an existing file"**.
3. اسحب (drag & drop) فولدر `heart_cad_app` كامل بمجلداته الفرعية (`views/`, `.streamlit/`)
   على المتصفح — المتصفحات الحديثة (Chrome/Firefox) بتحافظ على شكل الفولدرات تلقائياً.
4. اكتب commit message واضغط **Commit changes**.

### الطريقة 2 — عن طريق git من جهازك
```bash
cd heart_cad_app
git init
git add .
git commit -m "Initial commit: CardioSound AI Streamlit app"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```

---

## 5. النشر (Deploy) على Streamlit Community Cloud

1. روح [share.streamlit.io](https://share.streamlit.io) وسجّل دخول بحساب GitHub بتاعك.
2. **New app** → اختار الـ repo اللي رفعته.
3. Main file path: `app.py`
4. (اختياري) من **Advanced settings → Secrets**، تقدر تحط أي من دول لو عايز تغيّرهم عن
   القيم الافتراضية المظبوطة في `config.py`:
   ```toml
   GDRIVE_MODEL_FILE_ID = "..."
   GDRIVE_STATS_FILE_ID = "..."
   GDRIVE_SYNC_DATASET = "true"   # لتفعيل مزامنة فولدرات CWT/MFCC (قسم 3 فوق)
   ```
5. دوس **Deploy** — أول build هياخد شوية دقايق (تثبيت tensorflow-cpu وباقي المكتبات)،
   وأول فتح للتطبيق هياخد دقيقة إضافية لتنزيل الموديل من Drive.

---

## 6. اللي مش موجود في الـ repo ده عمداً

- **الموديل المدرَّب** (`.keras`) وملف **`mfcc_standardization_stats.npz`** — على Google
  Drive، بينزّلوا تلقائياً (قسم 2 فوق).
- **فولدرات CWT/MFCC الأصلية** — على Google Drive، مزامنتها اختيارية ومتوقفة افتراضياً
  (قسم 3 فوق).
- **نوتبوك التدريب** (`final_hybrid.ipynb`) وكود استخراج الـ MFCC/CWT من الصوت الخام —
  مش من ضمن الملفات اللي بعتّها، فمش موجودين هنا. لو عايز تضيفهم للـ repo كمرجع
  (من غير ما يتحطوا في مسار التشغيل الفعلي)، ابعتهم وأنا أضيفهم في فولدر منفصل
  زي `training/`.

---

## استكشاف الأخطاء

- **"Error installing requirements" / `No solution found... tensorflow-cpu... no wheels
  with a matching Python ABI tag`** → Streamlit Cloud شغّل التطبيق بنسخة Python حديثة
  جداً (3.14) لسه tensorflow-cpu مش بيدعمها. الحل: ثبّت نسخة Python على `3.11`
  (متوافقة مع tensorflow-cpu). فولدر المشروع في الـ ZIP ده فيه دلوقتي ملفين لضمان إن
  Streamlit Cloud ياخد بيهم مهما كانت الآلية اللي بيقرا بيها (`.python-version` و
  `runtime.txt`)، فمحتاجتش تعمل حاجة تانية لو رفعت المشروع من الأول من الـ ZIP ده.

  لو أنت رفعت الـ repo قبل ما أضيف الملفين دول، خطوات الإصلاح على الـ repo الموجود
  عندك بالفعل:
  1. على GitHub، في الـ repo بتاعك، دوس **Add file → Create new file**.
  2. اكتب اسم الملف `.python-version` (بالنقطة في الأول) وحط جواه سطر واحد: `3.11`
  3. Commit مباشرة على branch `main`. كرر نفس الخطوات لملف اسمه `runtime.txt` وجواه
     سطر واحد: `python-3.11`
  4. ارجع لـ Streamlit Cloud → **Manage app** → **Reboot app** (أو من قائمة الثلاث نقط
     اختار Reboot). هيعيد بناء البيئة بـ Python 3.11 والتثبيت هيعدي عادي.
- **"No trained model file was found"** → تأكد إن `GDRIVE_MODEL_FILE_ID` مظبوط صح في
  الـ Secrets، وإن صلاحية المشاركة على الملف "Anyone with the link".
- **التطبيق بطيء أول مرة** → طبيعي، ده وقت تنزيل الموديل (231 MB) + تحميله في الذاكرة.
  المرات اللي بعد كده هتبقى أسرع.
- **Predictions غريبة** → `audio_utils.py` بيستخدم إعدادات MFCC/CWT قياسية. لو
  الـ preprocessing الأصلي بتاعك (في `final_hybrid.ipynb`) كان مستخدم إعدادات مختلفة
  (wavelet, scales, colormap)، لازم تظبطهم في `audio_utils.py` عشان يطابقوا بالظبط.
