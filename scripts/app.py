from flask import Flask, request, jsonify
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
from tensorflow.keras.applications.convnext import preprocess_input

app = Flask(__name__)
API_PREFIX = "/api"

# =========================
# Models
# =========================
print(" Loading Places model...")
model1 = tf.keras.models.load_model('models/Luxora_Final_Model_v1.keras')
print("✅ Places model loaded!")

class_names_1 = [
    'Colossoi of Memnon',
    'Hatshepsut Temple',
    'Karnak Precinct',
    'Luxor Temple'
]

print(" Loading YOLO model...")
model2 = YOLO("models/best.pt")
print("✅ YOLO model loaded!")


# =========================
#  Normalize (IMPORTANT FIX)
# =========================
def normalize(text):
    return text.strip().lower()


def norm_set(items):
    """normalize full set safely"""
    return frozenset(normalize(x) for x in items)


# =========================
# 📚 Translations (English & Arabic)
# =========================
translations = {
    norm_set(['Ramses']): {
        'English': "Ramesses II, commonly known as Ramesses the Great, was the third pharaoh of the Nineteenth Dynasty of Egypt. Ramesses II is regarded as the greatest, most celebrated, and most powerful pharaoh of the New Kingdom, which itself was the most powerful period of ancient Egypt.",
        'Arabic': "رمسيس الثاني، المعروف باسم رمسيس العظيم، كان الفرعون الثالث من الأسرة التاسعة عشرة المصرية. يعتبر رمسيس الثاني أعظم وأشهر وأقوى فراعنة المملكة الجديدة، والتي كانت في حد ذاتها أقوى فترة في تاريخ مصر القديمة."
    },

    norm_set(['Ramses', 'Amun']): {
        'English': "In this ritual relief, Pharaoh Ramesses II is depicted in a pious stance, offering incense or libations to the supreme deity Amun-Ra. The king is often seen wearing his Khepresh (Blue Crown) and royal kilt, while Amun-Ra stands before him adorned in the iconic Shuti (double-plumed crown), symbolizing his role as the King of the Gods. This scene represents the divine exchange at the heart of Egyptian kingship: the pharaoh maintains Ma'at (universal order) through his devotion, and in return, the god bestows life, strength, and a prosperous reign upon the sovereign. The composition is framed by monumental statues and intricate cartouches, serving as a permanent stone record of the king's legitimacy and his direct connection to the divine.",
        'Arabic': "في هذا النقش الطقسي، يُصور الفرعون رمسيس الثاني في موقف توقد، يقدم البخور أو السكائب إلى الإله الأعظم آمون-را. غالبًا ما يُرى الملك يرتدي خبزته (التاج الأزرق) والفستان الملكي، بينما يقف آمون-را أمامه مرتديًا الشوتي الشهير (التاج ذي الرطوبتين المزدوجة)، مما يرمز إلى دوره كملك الآلهة. يمثل هذا المشهد التبادل الإلهي في قلب الملكية المصرية: يحافظ الفرعون على المعت (النظام الكوني) من خلال إخلاصه، وفي المقابل، يمنح الإله الحياة والقوة والسلطة المزدهرة للملك. يتأطر التركيب بتماثيل ضخمة وخرطوشات معقدة، مما يخدم كسجل حجري دائم لشرعية الملك وصلته المباشرة بالإلهي."
    },

    norm_set(['Ramses', 'Amun', 'Mut']): {
        'English': "In this scene, the composition expands to feature the Theban Triad, depicting Ramesses II in the presence of the divine couple, Amun-Ra and Mut. The Pharaoh is typically positioned at one end of the grouping, offering symbols of authority or food to Amun-Ra, who occupies the central or primary position with his tall, double-plumed crown. Standing behind or beside Amun is the goddess Mut, the Great Mother, recognizable by her vulture headdress. The interaction highlights a familial divine hierarchy; as Ramesses honors the (father) (Amun) and the (mother) (Mut), he reinforces his own status as their earthly (son) and legitimate heir.",
        'Arabic': "في هذا المشهد، يتوسع التركيب ليعرض الثلاثي الطيبي، يصور رمسيس الثاني في حضور الزوج الإلهي، آمون-را وموت. عادةً ما يتم وضع الفرعون في نهاية واحدة من المجموعة، يقدم رموز السلطة أو الطعام لآمون-را، الذي يشغل المركز أو المكان الرئيسي بتبوته الطويل ذي الرطوبتين المزدوجة. تقف الإلهة موت، الأم العظيمة، خلف أو بجانب آمون، يمكن التعرف عليها من خلال رأسها من النسر الأسود. يبرز التفاعل هرمية إلهية عائلية؛ بينما يكرم رمسيس (الأب) (آمون) و(الأم) (موت)، يقوي مركزه الخاص ك(ابن)هم الأرضي والوريث الشرعي."
    },

    norm_set(['Ramses', 'Amun', 'Horus']): {
        'English': "the scene depicts Ramesses II positioned between Amun-Ra and Horus, a composition that emphasizes the Pharaoh's divine legitimacy and his protection by the gods. Horus, the falcon-headed god of the sky, is typically shown wearing the Double Crown, while Amun-Ra stands with his characteristic twin-plumed headdress. Together, they are often portrayed in the act of blessing the King or leading him into the sanctuary, signaling his right to rule both the physical and spiritual realms.",
        'Arabic': "يصور المشهد رمسيس الثاني موضوعًا بين آمون-را وحورس، تركيب يبرز الشرعية الإلهية للفرعون وحمايته من الآلهة. عادةً ما يُصور حورس، إله السماء ذو رأس الصقر، يرتدي التاج المزدوج، بينما يقف آمون-را مع رأسه ذي الرطوبتين المزدوجة المتميزة. معًا، غالبًا ما يتم تصويرهم في عملية باركة الملك أو قيادته إلى المكان المقدس، مما يرمز إلى حقه في حكم العوالم المادية والروحية على حد سواء."
    },

    norm_set(['Alexander']): {
        'English': "In the Granite Sanctuary of Luxor Temple, Alexander the Great is depicted in traditional Pharaonic relief, a stylistic choice by the Macedonian king to solidify his legitimacy as a true Egyptian ruler. He is portrayed in the classic Pharaonic profile, often wearing the Khepresh or the Double Crown, standing in a ritualized posture before a table of offerings.",
        'Arabic': "في المكان المقدس من الجرانيت في معبد الأقصر، يُصور الإسكندر العظيم في نقش فرعوني تقليدي، اختيار أسلوبي من قبل الملك المقدوني لتثبيت شرعيته كحاكم مصري حقيقي. يُصور في المظهر الجانبي الفرعوني الكلاسيكي، غالبًا يرتدي الخبزة أو التاج المزدوج، يقف في موقف طقسي أمام مائدة العروض."
    },

    norm_set(['Alexander', 'Amun']): {
        'English': "In these reliefs, Alexander the Great is depicted performing sacred rites before Amun-Ra, often in his ithyphallic form as Amun-Min. Alexander is typically shown offering incense, flowers, or libations, dressed in the traditional Egyptian royal regalia to symbolize his status as the (Son of Amun.) The deity stands as the recipient of these gifts, characterized by his tall double-plumed crown and a raised arm holding a flail. This interaction serves as a visual bridge between Macedonian rule and Egyptian religious tradition, portraying the foreign conqueror as a pious protector of the ancient faith.",
        'Arabic': "في هذه النقوش، يُصور الإسكندر العظيم يؤدي الطقوس المقدسة أمام آمون-را، غالبًا في شكله الجسدي كآمون-مين. عادةً ما يُصور الإسكندر يقدم البخور أو الأزهار أو السكائب، مرتديًا الزي الملكي المصري التقليدي ليرمز إلى مركزه ك(ابن آمون). يقف الإله كمستقبل لهذه الهدايا، يتميز بتبوته الطويل ذي الرطوبتين المزدوجة وذراع مرتفع يحمل المضرب. يخدم هذا التفاعل كجسر بصري بين الحكم المقدوني والتقاليد الدينية المصرية، يصور الغازي الأجنبي كحامي توقد للديانة القديمة."
    },

    norm_set(['Alexander', 'Amun', 'Horus']): {
        'English': ": In this scene, Alexander the Great stands in the company of Amun-Ra and Horus, a grouping that reinforces his status as a divinely chosen Pharaoh. Alexander is depicted in traditional Egyptian royal attire, often positioned between the two deities to receive their joint blessing. Amun-Ra, identifiable by his tall double-plumed crown, represents the supreme source of royal power, while the falcon-headed Horus provides the celestial protection and legitimacy associated with the living King. Together, they are often shown placing their hands on Alexander or presenting him with the Ankh (symbol of life), visually integrating the Macedonian ruler into the ancient cycle of kingship.",
        'Arabic': "في هذا المشهد، يقف الإسكندر العظيم في صحبة آمون-را وحورس، تجميع يقوي مركزه كفرعون مختار إلهيًا. يُصور الإسكندر في زي ملكي مصري تقليدي، عادةً موضوعًا بين الإلهين لاستلام باركتهما المشتركة. يمثل آمون-را، الذي يمكن التعرف عليه من خلال تبوته الطويل ذي الرطوبتين المزدوجة، المصدر الأعلى للسلطة الملكية، بينما يوفر حورس ذو رأس الصقر الحماية السماوية والشرعية المرتبطة بالملك الحي. معًا، غالبًا ما يتم تصويرهم يضعون أيديهم على الإسكندر أو يقدمون له عنخ (رمز الحياة)، مما يدمج بصريًا الحاكم المقدوني في دورة الملكية القديمة."
    },

    norm_set(['Uniting Scene']): {
        'English': "The Uniting of the Two Lands, known as the Sema-Tawy scene, is a powerful symbolic composition frequently found on the thrones of colossal statues or the bases of temple walls. The scene features two figures—often the Nile god Hapy in duplicate, or a combination of deities like Horus and Seth—standing on either side of a central pole. They are shown tying together the long-stemmed lotus (representing Upper Egypt) and the papyrus (representing Lower Egypt) around a central sema hieroglyph, which itself translates to (union).",
        'Arabic': "توحيد البلدين، المعروف باسم مشهد سما-تاوي، هو تركيب رمزي قوي يوجد غالبًا على عروش التماثيل الضخمة أو قواعد جدران المعابد. يتميز المشهد بوجود شخصين—غالبًا إله النيل هابي في نسخة مزدوجة، أو مزيج من الآلهة مثل حورس وسيت—يقفان على جانبي عمود مركزي. يُصوران يربطان معًا اللوتس ذو الساق الطويلة (الذي يمثل مصر العليا) والبردي (الذي يمثل مصر السفلى) حول حرف هيروغليفي سما مركزي، والذي يترجم بنفسه إلى (الوحدة)."
    },

    norm_set(['Son of Ramses']): {
        'English': "The Sons of Ramesses scene is a grand, repetitive composition typically found on the exterior walls or pylons of temples like Luxor or the Ramesseum. It depicts a long procession of the Pharaoh's many sons, arranged in a strict, linear hierarchy based on their birth order. Each prince is shown in a standardized (walking) profile, wearing the Side-lock of Youth (a thick braid on the side of a shaven head) and holding a ceremonial fan or a scepter. Despite the uniform appearance of their bodies, each figure is individualized by a unique hieroglyphic inscription above or beside them, identifying them by name and title, such as the famous Prince Khaemwaset.",
        'Arabic': "مشهد أبناء رمسيس هو تركيب عظيم ومتكرر يوجد عادةً على الجدران الخارجية أو الأعمدة للمعابد مثل الأقصر أو الرمسية. يصور موكب طويل لأبناء الفرعون العددين، مرتبين في هرمية خطية صارمة بناءً على ترتيب ولادتهم. يُصور كل أمير في مظهر جانبي (مشي) موحد، يرتدي قفل الشعر الجانبي للشباب (ضفيرة سميكة على جانب الرأس المحلوق) ويمسك مروحة طقسية أو صولجان. على الرغم من المظهر الموحد لأجسادهم، يتم تخصيص كل شخص بنقش هيروغليفي فريد فوقه أو بجانبه، يحددهم بالاسم واللقب، مثل الأمير الشهير خمواس."
    },

    norm_set(['Asiatics']): {
        'English': "These reliefs represent the diverse populations of the Levant, including the Canaanites, Amorites, and Hittites, with whom Egypt frequently clashed during the 18th and 19th Dynasties. Historically, these scenes often commemorate major military campaigns, such as the Battle of Kadesh or the suppression of revolts in Retjenu. These captives symbolized the Pharaoh's success in securing the northern frontiers and the lucrative trade paths that brought cedar, silver, and lapis lazuli into Egypt. Unlike the southern campaigns which were often focused on resource extraction, the northern depictions reflected the complex geopolitical rivalry between the Egyptian Empire and the other great powers of the Near East, serving as a public record of the Pharaoh's role as the supreme defender against foreign encroachment.",
        'Arabic': "تمثل هذه النقوش السكان المتنوعين في بلاد الشام، بما في ذلك الكنعانيون والأموريون والحيثيون، الذين كشفت مصر معهم صراعات متكررة خلال الأسرة الثامنة عشرة والتاسعة عشرة. من الناحية التاريخية، غالبًا ما تخلد هذه المشاهد الحملات العسكرية الكبرى، مثل معركة قادش أو قمع الثورات في ريجنو. رمزت هذه الأسرى إلى نجاح الفرعون في تأمين الحدود الشمالية والطرق التجارية المربحة التي جلبت الأرز والفضة واللازورد إلى مصر. على عكس الحملات الجنوبية التي غالبًا ما تركزت على استخراج الموارد، عكست التصويرات الشمالية التنافس الجيوسياسي المعقد بين الإمبراطورية المصرية والقوى العظمى الأخرى في الشرق الأدنى، مما خدم كسجل عام لدور الفرعون كمدافع أعظم ضد الاقتحام الأجنبي."
    },

    norm_set(['Africans']): {
        'English': "These figures historically represent the people of Kush and various Nubian chiefdoms located to the south of Egypt. Their presence in temple reliefs served as a permanent political statement regarding the Pharaoh's control over the gold-rich regions of the south and the vital trade routes leading into sub-Saharan Africa. During the New Kingdom, particularly under Ramesses II, these depictions were less about recording a single battle and more about reinforcing the concept of the Pharaoh as the (Universal Overlord.) The inclusion of these captives on temple pylons functioned as a religious (magical) protection; by depicting the enemies of Egypt as eternally bound and subdued, the Egyptians believed they were ensuring the actual stability and security of their borders through divine decree.",
        'Arabic': "تمثل هذه الشخصيات من الناحية التاريخية شعب كوش والشيوخ النوبية المختلفة الموجودة جنوب مصر. خدم وجودهم في نقوش المعابد كبيان سياسي دائم بشأن سيطرة الفرعون على المناطق الغنية بالذهب في الجنوب والطرق التجارية الحيوية المؤدية إلى إفريقيا جنوب الصحراء. خلال المملكة الجديدة، وبشكل خاص تحت حكم رمسيس الثاني، كانت هذه التصويرات أقل من تسجيل معركة واحدة وأكثر من تعزيز مفهوم الفرعون ك(السيد العالمي.) خدم تضمين هذه الأسرى على أعمدة المعابد كحماية دينية (سحرية)؛ من خلال تصوير أعداء مصر مرتبطين ومخضعين إلى الأبد، اعتقد المصريون أنهم يضمنون الاستقرار والأمن الفعلي لحدودهم من خلال مرسوم إلهي."
    },

    norm_set(['Werethekau']): {
        'English': "This scene represents the ritual of divine legitimation, where the goddess Werethekau is depicted conferring royal authority upon the Pharaoh. Known as the (Great of Magic,) her primary historical role in temple iconography is to act as the protector and personification of the royal crowns. She is often shown in coronation sequences, where she stands behind or beside the King to facilitate the transition of power and ensure his physical and spiritual safety. By including Werethekau in this composition, the relief provides a formal record of the Pharaoh's divine right to rule, emphasizing that his sovereignty is sanctioned and maintained by the fundamental magical forces of the state.",
        'Arabic': "يمثل هذا المشهد طقس الشرعية الإلهية، حيث يُصور الإلهة وريثكاو تمنح السلطة الملكية للفرعون. المعروفة باسم(عظيمة السحر،) دورها التاريخي الرئيسي في أيقونات المعابد هو العمل كحامية وتجسيد للتاجات الملكية. غالبًا ما تُصور في تسلسلات التتويج، حيث تقف خلف أو بجانب الملك لتسهيل انتقال السلطة وتأمين سلامته الجسدية والروحية. من خلال تضمين وريثكاو في هذا التركيب، يوفر النقش سجلًا رسميًا لحق الفرعون الإلهي في الحكم، مؤكدًا أن سيادته تم إقرارها وتعزيزها من خلال القوى السحرية الأساسية للدولة."
    },

    norm_set(['Monkeys']): {
        'English': "This scene represents the solar cycle and the eternal rebirth of the sun, characterized by the presence of sacred baboons (often referred to as the (Adoring Baboons)) carved into the pedestal of the obelisk. Historically, these monkeys represent the (Eastern Souls) who, according to Egyptian belief, were the first to witness and hail the rising sun each morning. Their raised arms signify a gesture of worship and transformation, as their chattering at dawn was interpreted by the Egyptians as a ritual hymn to the god Ra as he emerged from the underworld. By placing these figures at the base of an obelisk—which itself is a stylized ray of sun—the relief anchors the monument in a perpetual state of solar adoration, ensuring that the Pharaoh's monument is spiritually revitalized at every sunrise.",
        'Arabic': "يمثل هذا المشهد الدورة الشمسية والولادة الأبدية للشمس، يتميز بوجود البابون المقدس (الذي يشار إليه غالبًا باسم(البابون المتعبد)) المحفور في قاعدة الأوبليسك. من الناحية التاريخية، تمثل هذه القرود(الأرواح الشرقية) التي، وفقًا للاعتقاد المصري، كانت الأولى التي شهدت وحيّدت الشمس المطلعة كل صباح. تشير أذرعهم المرتفعة إلى موقف عبادة وتحويل، حيث كان ضجيجهم عند الفجر يفسر المصريون كترنيمة طقسية لإله را بينما يظهر من العالم السفلي. من خلال وضع هذه الشخصيات في قاعدة الأوبليسك—الذي هو في حد ذاته شعاع شمس مصمم—يربط النقش النصب في حالة دائمة من عبادة الشمس، مما يضمن أن نصب الفرعون يتم تنشيطه روحيًا في كل شروق الشمس."
    }
}


# =========================
# Health Check Endpoint
# =========================
@app.route(f'{API_PREFIX}/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'places': model1 is not None,
            'yolo': model2 is not None
        }
    })

# =========================
# Route 1 → Places
# =========================
@app.route(f'{API_PREFIX}/predict', methods=['POST'])
def predict_places():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']

        img = Image.open(file).convert('RGB').resize((224, 224))
        img_array = np.array(img, dtype=np.float32)

        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model1.predict(img_array)[0]
        idx = int(np.argmax(predictions))

        # Return format compatible with Laravel Backend
        return jsonify({
            'success': True,
            'message': 'Image scanned successfully',
            'landmark_name': class_names_1[idx],
            'confidence': round(float(predictions[idx]) * 100, 2),
            'model': 'places',
            'prediction': class_names_1[idx]
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# =========================
# Route 2 → YOLO (with language support)
# =========================
@app.route(f'{API_PREFIX}/predict-yolo', methods=['POST'])
def predict_yolo():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        language = request.form.get('language', 'English')
        img = Image.open(io.BytesIO(file.read())).convert('RGB')

        results = model2.predict(source=img, verbose=False)

        detected_names = set()

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls.item())
                class_name = model2.names.get(class_id, f"class_{class_id}")
                detected_names.add(normalize(class_name))

        if not detected_names:
            no_figures = {
                'English': 'No figures detected.',
                'Arabic': 'لم يتم اكتشاف أي شخصيات.'
            }
            return jsonify({
                'model': 'yolo',
                'detected_objects': [],
                'translation': no_figures.get(language, no_figures['English'])
            })

        current_set = frozenset(detected_names)

        # =========================
        # PERFECT MATCH ONLY
        # =========================
        translation_data = translations.get(current_set)

        # =========================
        # SMART FALLBACK (subset match)
        # =========================
        if not translation_data:
            for key in translations:
                if key.issubset(current_set) or current_set.issubset(key):
                    translation_data = translations[key]
                    break

        # =========================
        # Get translation for requested language
        # =========================
        if translation_data:
            translation = translation_data.get(language, translation_data.get('English', ''))
        else:
            # FINAL FALLBACK
            info_texts = {
                'English': lambda name: f"Info about {name}.",
                'Arabic': lambda name: f"معلومات عن {name}."
            }
            translation_parts = []
            for name in detected_names:
                name_set = norm_set([name])
                if name_set in translations:
                    trans = translations[name_set].get(language, translations[name_set].get('English', ''))
                    translation_parts.append(trans)
                else:
                    translation_parts.append(info_texts.get(language, info_texts['English'])(name))
            translation = " ".join(translation_parts)

        return jsonify({
            'model': 'yolo',
            'detected_objects': list(detected_names),
            'translation': translation
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =========================
# Route 3 → Translation (with language support)
# =========================
@app.route(f'{API_PREFIX}/translate', methods=['POST'])
def translate_hieroglyphics():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        language = request.form.get('language', 'English')

        # Use YOLO model to detect hieroglyphic figures for proper translation
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        results = model2.predict(source=img, verbose=False)

        detected_names = set()

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls.item())
                class_name = model2.names.get(class_id, f"class_{class_id}")
                detected_names.add(normalize(class_name))

        if not detected_names:
            no_figures = {
                'English': 'No hieroglyphic figures detected in this image.',
                'Arabic': 'لم يتم اكتشاف أي شخصيات هيروغليفية في هذه الصورة.'
            }
            return jsonify({
                'model': 'translation',
                'detected_objects': [],
                'translation': no_figures.get(language, no_figures['English']),
                'confidence': 0.0
            })

        current_set = frozenset(detected_names)

        # =========================
        # PERFECT MATCH - Use the existing translations dictionary
        # =========================
        translation_data = translations.get(current_set)

        # =========================
        # SMART FALLBACK (subset match)
        # =========================
        if not translation_data:
            for key in translations:
                if key.issubset(current_set) or current_set.issubset(key):
                    translation_data = translations[key]
                    break

        # =========================
        # Get translation for requested language
        # =========================
        if translation_data:
            translation = translation_data.get(language, translation_data.get('English', ''))
        else:
            # FINAL FALLBACK - Individual translations
            figure_texts = {
                'English': lambda name: f"The figure {name} appears in this hieroglyphic text.",
                'Arabic': lambda name: f"الشخصية {name} تظهر في هذا النص الهيروغليفي."
            }
            individual_translations = []
            for name in detected_names:
                name_set = norm_set([name])
                if name_set in translations:
                    trans = translations[name_set].get(language, translations[name_set].get('English', ''))
                    individual_translations.append(trans)
                else:
                    individual_translations.append(figure_texts.get(language, figure_texts['English'])(name))
            translation = " ".join(individual_translations)

        # Calculate confidence based on detection confidence
        confidence = 0.0
        if results and results[0].boxes is not None:
            confidences = [float(box.conf[0]) for box in results[0].boxes]
            confidence = max(confidences) if confidences else 0.0

        original_texts = {
            'English': f"Hieroglyphic text containing: {', '.join(detected_names)}",
            'Arabic': f"نص هيروغليفي يحتوي على: {', '.join(detected_names)}"
        }

        return jsonify({
            'model': 'translation',
            'detected_objects': list(detected_names),
            'translation': translation,
            'confidence': round(confidence * 100, 2),
            'original_text': original_texts.get(language, original_texts['English'])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =========================
# Run
# =========================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
