import streamlit as st
import onnxruntime as ort
from PIL import Image
import numpy as np

# إعداد واجهة التطبيق
st.set_page_config(page_title="Arabic Sign Language Translator", layout="centered")
st.title("🤟 Arabic Sign Language Translator")
st.write("Take a photo or upload an image of an Arabic sign letter to translate it!")

# 1. تحميل النموذج (ONNX)
@st.cache_resource
def load_model():
    # تأكد أن ملف الـ ONNX موجود في نفس مجلد ملف الـ app.py
    return ort.InferenceSession("sign_language_model.onnx")

session = load_model()

# 2. قائمة الحروف العربية (نفس الترتيب المستخدم في التدريب)
classes = ['ألف', 'باء', 'تاء', 'ثاء', 'جيم', 'حاء', 'خاء', 'دال', 'ذال', 'راء', 'زاي', 'سين', 'شين', 'صاد', 'ضاد', 'طاء', 'ظاء', 'عين', 'غين', 'فاء', 'قاف', 'كاف', 'لام', 'ميم', 'نون', 'هاء', 'واو', 'ياء']

# 3. اختيار طريقة الإدخال (كاميرا أو رفع ملف)
input_method = st.radio("Select input method:", ("Camera", "Upload File"))

img_file = None
if input_method == "Camera":
    img_file = st.camera_input("Capture sign")
else:
    img_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# 4. معالجة التوقع إذا تم توفير صورة
if img_file is not None:
    # فتح الصورة وتحويلها لرمادي (Grayscale)
    image = Image.open(img_file).convert('L') 
    st.image(image, caption='Processed Image', width=300)
    
    # 5. معالجة الصورة لتناسب مدخلات النموذج (64x64)
    img = image.resize((64, 64))
    img_array = np.array(img).astype(np.float32)
    
    # الـ Normalize (نفس المستخدم في Notebook تماماً)
    img_array = (img_array / 255.0 - 0.5) / 0.5  
    
    # إعادة تشكيل المصفوفة لتناسب أبعاد النموذج (Batch=1, Channel=1, H=64, W=64)
    img_array = img_array.reshape(1, 1, 64, 64) 

    # 6. التوقع (Inference)
    st.write("---")
    st.write("🔍 **Analyzing...**")
    
    inputs = {session.get_inputs()[0].name: img_array}
    outputs = session.run(None, inputs)
    prediction = np.argmax(outputs[0])
    
    # 7. عرض النتيجة النهائية بشكل مميز
    st.balloons() # تأثير احتفالي بسيط عند النجاح
    st.success(f"### The predicted letter is: **{classes[prediction]}**")