import os, tempfile, requests, logging, asyncio, re, hashlib
from datetime import datetime
from typing import List, Dict, Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters, ConversationHandler, CallbackQueryHandler

# ================== إعدادات محسنة ==================
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_LINES = 200000
CHUNK_SIZE = 5000
SUPPORTED_LARGE_EXTENSIONS = {'.py', '.js', '.java', '.cpp', '.c', '.txt', '.log', '.csv', '.json', '.xml', '.html', '.css', '.php', '.rb', '.go', '.rs', '.ts', '.sql'}

# ================== حالات المحادثة ==================
WAITING_TOKEN, WAITING_FILE, WAITING_INSTRUCTIONS, WAITING_BACKEND_CHOICE = range(4)

# ================== نماذج محسنة مع توكنات فعلية ==================
BACKEND_KEYS = {
    "KIMI": {
        "key": "sk-your-kimi-token-here",  # 🔹 استبدل بالتوكين الفعلي
        "base": "https://api.moonshot.cn/v1",
        "models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
        "context": 128000
    },
    "DEEPSEEK": {
        "key": "sk-your-deepseek-token-here",  # 🔹 استبدل بالتوكين الفعلي
        "base": "https://api.deepseek.com",
        "models": ["deepseek-chat", "deepseek-coder"],
        "context": 64000
    },
    "CHATGPT": {
        "key": "sk-your-openai-token-here",  # 🔹 استبدل بالتوكين الفعلي
        "base": "https://api.openai.com",
        "models": ["gpt-4", "gpt-3.5-turbo"],
        "context": 128000
    }
}

user_sessions = {}
file_history = {}

# ================== أدوات مساعدة محسنة ==================
def validate_token(token: str) -> bool:
    """التحقق من صحة توكن البوت"""
    pattern = r'^\d+:[A-Za-z0-9_-]+$'
    return bool(re.match(pattern, token))

def sanitize_filename(filename: str) -> str:
    """تنظيف اسم الملف من الأحرف الخطرة"""
    return re.sub(r'[^\w\-_.]', '_', filename)

def detect_language(filename: str) -> str:
    """كشف لغة البرمجة من امتداد الملف"""
    ext_lang_map = {
        '.py': 'Python', '.js': 'JavaScript', '.java': 'Java',
        '.cpp': 'C++', '.c': 'C', '.html': 'HTML', '.css': 'CSS',
        '.php': 'PHP', '.rb': 'Ruby', '.go': 'Go', '.rs': 'Rust',
        '.ts': 'TypeScript', '.sql': 'SQL', '.json': 'JSON',
        '.xml': 'XML', '.csv': 'CSV', '.txt': 'Text'
    }
    ext = os.path.splitext(filename)[1].lower()
    return ext_lang_map.get(ext, 'نص عادي')

def create_backup(user_id: int, filename: str, content: str) -> str:
    """إنشاء نسخة احتياطية محسنة"""
    backup_dir = f"backups/{user_id}"
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = sanitize_filename(filename)
    backup_file = f"{backup_dir}/{safe_filename}_{timestamp}.bak"
    
    try:
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        return backup_file
    except Exception as e:
        logging.error(f"Backup failed: {e}")
        return ""

def validate_backend_keys() -> Dict[str, bool]:
    """التحقق من صحة توكنات النماذج"""
    validation_results = {}
    
    for name, config in BACKEND_KEYS.items():
        if config["key"].startswith("sk-your-"):
            validation_results[name] = False
            logging.warning(f"❌ {name}: توكن غير مضبوط")
        else:
            validation_results[name] = True
            logging.info(f"✅ {name}: التوكن جاهز")
    
    return validation_results

# ================== نظام استدعاء النماذج المحسن ==================
async def call_backend_advanced(name: str, filename: str, text: str, instructions: str, model: str = None) -> Dict:
    """استدعاء متقدم محسن لنماذج الذكاء الاصطناعي"""
    if name not in BACKEND_KEYS:
        return {
            "success": False,
            "error": f"النموذج {name} غير مدعوم",
            "content": f"❌ النموذج {name} غير مدعوم"
        }
    
    info = BACKEND_KEYS[name]
    
    # التحقق من التوكن
    if info["key"].startswith("sk-your-"):
        return {
            "success": False,
            "error": f"توكن {name} غير مضبوط",
            "content": f"❌ يرجى ضبط توكن {name} أولاً"
        }
    
    if model is None:
        model = info["models"][0]
    
    # إصلاح مسار API
    if name == "CHATGPT":
        url = f"{info['base']}/v1/chat/completions"
    else:
        url = f"{info['base']}/chat/completions"
    
    system_prompt = f"""أنت مساعد برمجي خبير. الملف: {filename} - اللغة: {detect_language(filename)}

التعليمات: {instructions}

المطلوب:
1. تحليل الملف الحالي وفهم هيكله
2. تطبيق التعليمات بدقة عالية
3. الحفاظ على نمط البرمجة الأصلي
4. إضافة تعليقات توضيحية عند الحاجة
5. التأكد من صحة الشيفرة من الناحية الوظيفية
6. اقتراح تحسينات إضافية ذات صلة

أعد الشيفرة المحسنة فقط بدون شرح إضافي:"""
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"```{detect_language(filename).lower()}\n{text}\n```"}
        ],
        "temperature": 0.1,
        "max_tokens": min(info["context"] - 1000, 32000),
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1
    }
    
    headers = {
        "Authorization": f"Bearer {info['key']}", 
        "Content-Type": "application/json",
        "User-Agent": "AdvancedCodeBot/3.0"
    }
    
    try:
        async with asyncio.timeout(120):  # زيادة الوقت لمناولة الملفات الكبيرة
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # استخراج الشيفرة من markdown code blocks إذا وجدت
            code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', content, re.DOTALL)
            if code_blocks:
                content = code_blocks[0]
            else:
                # إذا لم توجد code blocks، استخدم المحتوى كما هو
                content = content.strip()
            
            return {
                "success": True,
                "content": content,
                "model_used": model,
                "tokens_used": result.get("usage", {}).get("total_tokens", 0)
            }
    except asyncio.TimeoutError:
        logging.error(f"Timeout calling {name}")
        return {
            "success": False,
            "error": "انتهت مهلة الطلب",
            "content": f"❌ {name}: انتهت مهلة الطلب (120 ثانية)"
        }
    except Exception as e:
        logging.error(f"Error calling {name}: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "content": f"❌ خطأ في {name}: {str(e)}"
        }

# ================== معالجة الملفات الكبيرة المحسنة ==================
class AdvancedFileProcessor:
    def __init__(self):
        self.chunks = []
        self.current_chunk = 0
    
    def split_file(self, content: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
        """تقسيم الملف إلى أجزاء صغيرة مع الحفاظ على السياق"""
        lines = content.split('\n')
        total_lines = len(lines)
        
        chunks = []
        for i in range(0, total_lines, chunk_size):
            # إضافة تداخل بسيط بين الأجزاء للحفاظ على السياق
            start = max(0, i - 10)  # 10 أسطر سابقة للسياق
            end = min(total_lines, i + chunk_size + 10)  # 10 أسطر لاحقة للسياق
            
            chunk = '\n'.join(lines[start:end])
            chunks.append({
                'content': chunk,
                'start_line': start + 1,
                'end_line': end,
                'is_context': i != start
            })
        
        return chunks
    
    def analyze_file_complexity(self, content: str) -> Dict:
        """تحليل تعقيد الملف"""
        lines = content.split('\n')
        total_lines = len(lines)
        total_chars = len(content)
        
        # تحليل التعقيد الأساسي
        code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        comment_lines = [line for line in lines if line.strip().startswith('#') or '//' in line]
        
        return {
            "total_lines": total_lines,
            "total_chars": total_chars,
            "code_lines": len(code_lines),
            "comment_lines": len(comment_lines),
            "comment_ratio": len(comment_lines) / max(total_lines, 1),
            "estimated_chunks": (total_lines + CHUNK_SIZE - 1) // CHUNK_SIZE,
            "avg_line_length": total_chars / max(total_lines, 1),
            "file_hash": hashlib.md5(content.encode()).hexdigest()[:12],
            "complexity": "عالية" if total_lines > 10000 else "متوسطة" if total_lines > 1000 else "منخفضة"
        }

# ================== واجهة مستخدم محسنة ==================
async def start_advanced(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """بدء محادثة متقدمة محسنة"""
    user = update.effective_user
    
    # التحقق من توفر النماذج
    available_backends = validate_backend_keys()
    active_backends = [name for name, valid in available_backends.items() if valid]
    
    if not active_backends:
        await update.message.reply_text(
            "❌ **لا توجد نماذج متاحة حالياً**\n\n"
            "يجب ضبط توكنات النماذج في الإعدادات أولاً.\n"
            "توكنات النماذج الحالية غير مضبوطة بشكل صحيح.",
            parse_mode='Markdown'
        )
        return ConversationHandler.END
    
    # إنشاء واجهة اختيار النماذج
    keyboard = []
    for backend in active_backends:
        emoji = "✅" if available_backends[backend] else "❌"
        keyboard.append([InlineKeyboardButton(f"{emoji} {backend}", callback_data=f"backend_{backend}")])
    
    if len(active_backends) > 1:
        keyboard.append([InlineKeyboardButton("🚀 جميع النماذج المتاحة", callback_data="backend_all")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        f"🛠️ **مرحباً {user.first_name} في بوت التطوير المتقدم!**\n\n"
        f"📊 **النماذج المتاحة ({len(active_backends)}):**\n"
        f"اختر نموذج الذكاء الاصطناعي:",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )
    return WAITING_BACKEND_CHOICE

async def receive_file_enhanced(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """استقبال الملفات محسّن مع تحليل تلقائي"""
    msg = update.message
    user_id = update.effective_user.id
    
    if not msg.document:
        await msg.reply_text("❌ يرجى إرسال الملف كـ Document.")
        return WAITING_FILE
    
    file_size = msg.document.file_size or 0
    
    if file_size > MAX_FILE_SIZE:
        await msg.reply_text(
            f"❌ حجم الملف كبير جداً! ({file_size//1024//1024}MB)\n"
            f"الحد الأقصى: {MAX_FILE_SIZE//1024//1024}MB"
        )
        return WAITING_FILE
    
    filename = msg.document.file_name or "unnamed_file"
    file_ext = os.path.splitext(filename)[1].lower()
    
    # تحليل متقدم للملف
    processor = AdvancedFileProcessor()
    
    try:
        # تحميل الملف مع متابعة التقدم
        progress_msg = await msg.reply_text("📥 **جاري تحميل الملف...**")
        
        file = await ctx.bot.get_file(msg.document.file_id)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, sanitize_filename(filename))
            await file.download_to_drive(custom_path=path)
            
            await progress_msg.edit_text("🔍 **جاري تحليل الملف...**")
            
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                original_content = fh.read()
        
        # تحليل متقدم للملف
        analysis = processor.analyze_file_complexity(original_content)
        
        # حفظ في الجلسة
        user_sessions[user_id]["file"] = {
            "filename": filename,
            "content": original_content,
            "language": detect_language(filename),
            "size": len(original_content),
            "lines": analysis["total_lines"],
            "backup_path": create_backup(user_id, filename, original_content),
            "analysis": analysis,
            "complexity": analysis["complexity"]
        }
        
        file_info = user_sessions[user_id]["file"]
        
        # إرسال تحليل مفصل
        response_text = f"""
🎉 **تم استلام الملف بنجاح!** {'🚀' if analysis['complexity'] == 'عالية' else '📊'}

📄 **الملف:** `{filename}`
🔤 **اللغة:** {file_info['language']}
📈 **الحجم:** {analysis['total_lines']:,} سطر • {analysis['total_chars']:,} حرف
🏷️ **التعقيد:** {analysis['complexity']}
🧮 **البصمة:** `{analysis['file_hash']}`

📊 **تحليل إضافي:**
• أسطر كود: {analysis['code_lines']:,}
• أسطر تعليقات: {analysis['comment_lines']:,}
• نسبة التعليقات: {analysis['comment_ratio']:.1%}

💡 **اكتب الآن التعليمات التي تريد تطبيقها على الملف:**
        """
        
        if analysis["total_lines"] > 10000:
            response_text += f"\n⏳ **ملاحظة:** الملف كبير جداً، المعالجة قد تستغرق عدة دقائق."
        
        await progress_msg.edit_text(response_text, parse_mode='Markdown')
        return WAITING_INSTRUCTIONS
        
    except Exception as e:
        logging.error(f"File processing error: {e}")
        await update.message.reply_text(f"❌ خطأ في معالجة الملف: {str(e)}")
        return WAITING_FILE

# ================== أوامر إضافية محسنة ==================
async def setup_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """عرض إعدادات النماذج الحالية"""
    validation = validate_backend_keys()
    
    setup_text = "⚙️ **إعدادات النماذج الحالية:**\n\n"
    
    for name, config in BACKEND_KEYS.items():
        status = "✅ جاهز" if validation[name] else "❌ يحتاج إعداد"
        models = ", ".join(config["models"])
        
        setup_text += f"**{name}:** {status}\n"
        setup_text += f"   • النماذج: {models}\n"
        setup_text += f"   • السياق: {config['context']:,} رمز\n\n"
    
    setup_text += "🔧 لضبط التوكنات، قم بتعديل ملف البوت مباشرة."
    
    await update.message.reply_text(setup_text, parse_mode='Markdown')

async def help_enhanced(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """تعليمات محسنة"""
    help_text = """
🛠️ **بوت التطوير المتقدم - التعليمات**

📁 **الميزات:**
• معالجة ملفات حتى 50MB
• دعم 3+ نماذج ذكاء اصطناعي
• نسخ احتياطي تلقائي
• تحليل تلقائي للملفات

🎯 **الأوامر المتاحة:**
/start - بدء جلسة جديدة
/setup - عرض إعدادات النماذج  
/stats - عرض إحصائيات الجلسة
/help - عرض هذه التعليمات

📝 **طريقة الاستخدام:**
1. ابدأ بـ /start
2. اختر نموذج الذكاء الاصطناعي
3. أرسل الملف كـ Document
4. اكتب التعليمات المطلوبة
5. استلم الملف المحسن

🔧 **الملفات المدعومة:** Python, JS, Java, C++, وغيرها
    """
    
    await update.message.reply_text(help_text, parse_mode='Markdown')

# ================== التشغيل الرئيسي المحسن ==================
def main_enhanced():
    """تشغيل البوت المحسن"""
    
    # إنشاء الهيكل التنظيمي
    os.makedirs("backups/large_files", exist_ok=True)
    os.makedirs("temp/large_processing", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 🔹 إعداد التوكن الرئيسي
    BOT_TOKEN = "8214334664:AAGWEhTYrFTyN_TCxbFlQfdnIKYLgfI496A"
    
    # 🔹 التحقق من التوكنات المتاحة
    available_models = validate_backend_keys()
    active_models = [name for name, valid in available_models.items() if valid]
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/bot_enhanced.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    if not active_models:
        logger.warning("⚠️  لا توجد نماذج ذكاء اصطناعي مضبوطة!")
    else:
        logger.info(f"🚀 النماذج الجاهزة: {', '.join(active_models)}")
    
    app = ApplicationBuilder()\
        .token(BOT_TOKEN)\
        .concurrent_updates(True)\
        .pool_timeout(120)\
        .read_timeout(120)\
        .write_timeout(120)\
        .build()

    # معالج المحادثة المحسن
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start_advanced)],
        states={
            WAITING_BACKEND_CHOICE: [CallbackQueryHandler(backend_button_handler, pattern="^backend_")],
            WAITING_FILE: [MessageHandler(filters.Document.ALL, receive_file_enhanced)],
            WAITING_INSTRUCTIONS: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_instructions_super)]
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        name="enhanced_conv",
        allow_reentry=True
    )

    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("setup", setup_command))
    app.add_handler(CommandHandler("help", help_enhanced))
    app.add_handler(CommandHandler("start", start_advanced))

    logger.info("🚀 بدء تشغيل البوت المحسن للملفات العملاقة...")
    
    try:
        app.run_polling()
    except Exception as e:
        logger.error(f"❌ فشل تشغيل البوت: {e}")
        raise

if __name__ == "__main__":
    main_enhanced()
