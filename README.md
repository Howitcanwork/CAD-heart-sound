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
├── models/                            # فاضي عمداً - الموديل بينزّل تلقائياً من Hugging Face
├── assets/
├── .streamlit/config.toml              # ثيم التطبيق
└── requirements.txt
```

> ملحوظة: النوتبوك الأصلي اللي بعتّه (`Last_CAD_website_nns.ipynb`) هو Colab runner
> كان بيكتب نفس الملفات دي بأمر `%%writefile` ثم يشغّل التطبيق عبر tunnel. الملفات هنا
> هي نفس الكود لكن مستخرج كملفات حقيقية جاهزة لـ GitHub، مش نوتبوك.

---

## 2. الموديل المدرَّب (231 MB) — مربوط بـ Hugging Face بالفعل ✅

GitHub بيرفض أي ملف أكبر من 100 MB، فالموديل `final_model.keras` **ما بيتراپع للـ repo
خالص**. بدل كده، `model_utils.py` بينزّله تلقائياً من الـ Hugging Face Hub repo بتاعكم:

**[huggingface.co/noonatu/cad-heart-sound](https://huggingface.co/noonatu/cad-heart-sound)**

أول مرة التطبيق يشتغل (ويحتفظ بيه بعد كده)، باستخدام مكتبة `huggingface_hub` الرسمية
(أوثق وأسرع بكتير من Google Drive - مفيش صفحات تحذير أو quota).

الإعدادات متظبطة بالفعل جوه `config.py`:

```python
HF_REPO_ID = "noonatu/cad-heart-sound"
HF_MODEL_FILENAME = "final_model.keras"                     # 242 MB
HF_STATS_FILENAME = "mfcc_standardization_stats.npz"
```

فمفيش خطوة إضافية مطلوبة منك — بس تأكد إن الـ repo لسه **Public** وقت الـ deploy (لو
غيّرته لـ Private، محتاج تضيف `HF_TOKEN` في الـ Secrets - قسم 5 تحت).

> لو عايز تستخدم repo تاني (فورك للمشروع، أو حساب مختلف)، غيّر `HF_REPO_ID` في
> `config.py`، أو حطه كـ Secret في Streamlit Cloud (الـ Secrets دايماً بتلغي القيمة
> اللي في الكود).

---

## 3. الداتاسيت الأصلي (CWT + MFCC) — شغّال افتراضياً ✅

دي متخزنة في **repo تاني منفصل** (Dataset repo، مش Model repo):

**[huggingface.co/datasets/noonatu/cad-heart-sound-dataset](https://huggingface.co/datasets/noonatu/cad-heart-sound-dataset)**

وفيه فولدرين:
- `CWT_Scalogram_new299` (صور الـ CWT scalograms)
- `MFCC_Features_npy` (ملفات MFCC الخام)

كل واحد فيهم مقسّم لـ 5 فولدرات فرعية بأسماء الكلاسات بالظبط (`Normal`, `MVP`, `MR`, `MS`,
`AS`)، وأسماء الملفات جوّاهم شكلها `New_MVP_001.npy` / `New_MVP_001.png` وهكذا.

> **مهم عشان تختبر صح:** المطابقة بتتم بمقارنة اسم الملف اللي بترفعه (من غير الامتداد)
> باسم الملف في الداتاسيت **حرفياً**. يعني لو عايز تتأكد إن ground truth هيظهر، لازم
> ترفع ملف `.wav` اسمه بالظبط زي عينة موجودة، مثلاً `New_MVP_001.wav` (مش مجرد
> "mvp.wav" أو اسم تاني عشوائي).

الفولدرات دي بتُستخدم في `dataset_lookup.py`: لو المستخدم رفع ملف `.wav` هو نفسه موجود
ضمن عينات التدريب، التطبيق يجيب الـ features الحقيقية المحسوبة مسبقاً بدل ما يعيد حسابها
من الصوت الخام - وده اللي بيخلي النتيجة مطابقة تماماً لما كانت عليه في Colab.

**ليه شغّالة افتراضياً هنا (عكس إعداد Google Drive القديم)؟** لأن Hugging Face Hub
مصمم أصلاً لاستضافة ملفات وفولدرات ML كبيرة، فمفيهوش مشاكل الـ quota أو صفحات التحذير
اللي كانت بتحصل مع Drive. التحميل بيحصل مرة واحدة بس لكل نسخة شغّالة من التطبيق
(`st.cache_resource`)، والمرات اللي بعد كده بتبقى فورية.

لو حابب توقفها (مثلاً لو الفولدرات كبرت جداً وبقت بتاخد وقت طويل)، ضيف الـ Secret ده في
Streamlit Cloud:
```toml
HF_SYNC_DATASET = "false"
```
وقتها التطبيق هيرجع يحسب MFCC/CWT من الصوت المرفوع مباشرة (الطريقة القياسية).

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
   HF_REPO_ID = "..."             # لو هتستخدم repo تاني
   HF_SYNC_DATASET = "false"       # لإيقاف مزامنة فولدرات CWT/MFCC (قسم 3 فوق)
   HF_TOKEN = "..."                 # لازم بس لو الـ HF repo بقى Private
   ```
5. دوس **Deploy** — أول build هياخد شوية دقايق (تثبيت tensorflow-cpu وباقي المكتبات)،
   وأول فتح للتطبيق هياخد دقيقة أو اتنين إضافية لتنزيل الموديل + فولدرات الداتاسيت من
   Hugging Face.

---

## 6. اللي مش موجود في الـ repo ده عمداً

- **الموديل المدرَّب** (`.keras`)، ملف **`mfcc_standardization_stats.npz`**، وفولدرات
  **CWT/MFCC الأصلية** — كلهم على Hugging Face
  ([noonatu/cad-heart-sound](https://huggingface.co/noonatu/cad-heart-sound))، بينزّلوا
  تلقائياً (قسم 2 و 3 فوق).
- **نوتبوك التدريب** (`final_hybrid.ipynb`) وكود استخراج الـ MFCC/CWT من الصوت الخام —
  مش من ضمن الملفات اللي بعتّها لي هنا في المحادثة، فمش موجودين في نسخة الكود دي. لو
  رفعتهم كمان على نفس الـ Hugging Face repo، مش لازمين لتشغيل الموقع (بيفضل يشتغل عادي
  من غيرهم)، لكن لو عايزهم كمرجع جوه الـ GitHub repo نفسه، ابعتلي وأضيفهم في فولدر
  منفصل زي `training/`.

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
- **"No trained model file was found"** → تأكد إن `HF_REPO_ID` مظبوط صح (في `config.py`
  أو الـ Secrets)، وإن الـ repo على Hugging Face لسه **Public** (لو بقى Private، لازم
  تضيف `HF_TOKEN` في الـ Secrets).
- **التطبيق بطيء أول مرة** → طبيعي، ده وقت تنزيل الموديل (231 MB) + فولدرات CWT/MFCC (لو
  `HF_SYNC_DATASET` شغّالة) + تحميل الموديل في الذاكرة. المرات اللي بعد كده هتبقى أسرع
  بكتير.
- **Predictions غريبة لعينة موجودة في الداتاسيت الأصلي** → تأكد إن `HF_SYNC_DATASET`
  شغّالة (هي شغّالة افتراضياً)، وإن اسم الملف اللي بترفعه للموقع مطابق حرفياً (نفس الـ
  stem، من غير الامتداد) لاسم الملف في فولدر `CWT_Scalogram_new299` / `MFCC_Features_npy`
  على Hugging Face.
- **Predictions غريبة لتسجيل جديد مش موجود في الداتاسيت** → `audio_utils.py` بيستخدم
  إعدادات MFCC/CWT قياسية (تقريبية). لو الـ preprocessing الأصلي بتاعك (في
  `final_hybrid.ipynb`) كان مستخدم إعدادات مختلفة (wavelet, scales, colormap)، لازم
  تظبطهم في `audio_utils.py` عشان يطابقوا بالظبط.
