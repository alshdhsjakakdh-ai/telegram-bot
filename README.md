import os, tempfile, requests, logging, asyncio, re, hashlib
from datetime import datetime
from typing import List, Dict, Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters, ConversationHandler, CallbackQueryHandler

# ================== إعدادات محسنة ==================
MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_LINES = 200000
CHUNK_SIZE = 5000

# ================== توكنات نماذج الذكاء الاصطناعي ==================
BACKEND_KEYS = {
    "KIMI": {
        "key": "sk-your-kimi-token-here",
        "base": "https://api.moonshot.cn/v1",
        "models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
        "context": 128000
    },
    "DEEPSEEK": {
        "key": "sk-your-deepseek-token-here", 
        "base": "https://api.deepseek.com",
        "models": ["deepseek-chat", "deepseek-coder"],
        "context": 64000
    },
    "CHATGPT": {
        "key": "sk-your-openai-token-here",
        "base": "https://api.openai.com/v1",
        "models": ["gpt-4", "gpt-3.5-turbo"],
        "context": 128000
    },
    "CLAUDE": {
        "key": "sk-your-claude-token-here",
        "base": "https://api.anthropic.com/v1",
        "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229"],
        "context": 200000
    },
    "GEMINI": {
        "key": "sk-your-gemini-token-here",
        "base": "https://generativelanguage.googleapis.com/v1",
        "models": ["gemini-pro", "gemini-pro-vision"],
        "context": 32768
    },
    "QIANWEN": {
        "key": "sk-your-qianwen-token-here",
        "base": "https://dashscope.aliyuncs.com/api/v1",
        "models": ["qwen-turbo", "qwen-plus", "qwen-max"],
        "context": 128000
    },
    "ZHIPU": {
        "key": "sk-your-zhipu-token-here", 
        "base": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["glm-4", "glm-3-turbo"],
        "context": 128000
    }
}

user_sessions = {}
file_history = {}

# ================== نظام الدمج المتقدم ==================
class AdvancedMerger:
    def __init__(self):
        self.versions = []
        self.consensus_threshold = 0.7
    
    async def create_multiple_versions(self, filename: str, content: str, instructions: str, models: List[str]) -> List[Dict]:
        """إنشاء نسخ متعددة باستخدام نماذج مختلفة"""
        tasks = []
        for model_name in models:
            task = asyncio.create_task(
                self.generate_version(model_name, filename, content, instructions)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_versions = []
        for i, result in enumerate(results):
            if isinstance(result, dict) and result.get("success"):
                valid_versions.append({
                    "model": models[i],
                    "content": result["content"],
                    "quality_score": self.assess_quality(result["content"], content),
                    "tokens_used": result.get("tokens_used", 0)
                })
        
        return valid_versions
    
    async def generate_version(self, model_name: str, filename: str, content: str, instructions: str) -> Dict:
        """إنشاء نسخة باستخدام نموذج محدد"""
        if model_name not in BACKEND_KEYS:
            return {"success": False, "error": f"النموذج {model_name} غير مدعوم"}
        
        info = BACKEND_KEYS[model_name]
        
        if model_name == "CHATGPT":
            return await self.call_openai_api(info, filename, content, instructions)
        elif model_name == "CLAUDE":
            return await self.call_claude_api(info, filename, content, instructions)
        elif model_name == "GEMINI":
            return await self.call_gemini_api(info, filename, content, instructions)
        else:
            return await self.call_standard_api(info, filename, content, instructions, model_name)
    
    async def merge_versions(self, versions: List[Dict], original_content: str, instructions: str) -> Dict:
        """دمج النسخ المتعددة واختيار الأفضل"""
        if not versions:
            return {"success": False, "error": "لا توجد نسخ للدمج"}
        
        if len(versions) == 1:
            return {"success": True, "content": versions[0]["content"], "method": "single"}
        
        # تحليل التوافق بين النسخ
        compatibility_scores = self.analyze_compatibility(versions)
        
        # اختيار النسخة الأفضل جودة
        best_version = max(versions, key=lambda x: x["quality_score"])
        
        if compatibility_scores["consensus_achieved"]:
            # إذا كانت النسخ متوافقة، ندمجها
            merged_content = await self.smart_merge(versions, original_content, instructions)
            return {
                "success": True, 
                "content": merged_content,
                "method": "consensus_merge",
                "models_used": [v["model"] for v in versions],
                "compatibility_score": compatibility_scores["average"]
            }
        else:
            # إذا لم تكن متوافقة، نستخدم النسخة الأفضل
            return {
                "success": True,
                "content": best_version["content"],
                "method": "best_quality",
                "selected_model": best_version["model"],
                "quality_score": best_version["quality_score"]
            }
    
    def analyze_compatibility(self, versions: List[Dict]) -> Dict:
        """تحليل مدى توافق النسخ مع بعضها"""
        if len(versions) < 2:
            return {"consensus_achieved": True, "average": 1.0}
        
        scores = []
        for i in range(len(versions)):
            for j in range(i + 1, len(versions)):
                similarity = self.calculate_similarity(
                    versions[i]["content"], 
                    versions[j]["content"]
                )
                scores.append(similarity)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        consensus = avg_score >= self.consensus_threshold
        
        return {
            "consensus_achieved": consensus,
            "average": avg_score,
            "min": min(scores) if scores else 0,
            "max": max(scores) if scores else 0
        }
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """حساب درجة التشابه بين نصين"""
        lines1 = set(text1.split('\n'))
        lines2 = set(text2.split('\n'))
        
        if not lines1 and not lines2:
            return 1.0
        
        intersection = len(lines1.intersection(lines2))
        union = len(lines1.union(lines2))
        
        return intersection / union if union > 0 else 0
    
    def assess_quality(self, new_content: str, original_content: str) -> float:
        """تقييم جودة المحتوى الجديد"""
        original_lines = original_content.split('\n')
        new_lines = new_content.split('\n')
        
        if not original_lines or not new_lines:
            return 0.5
        
        # حساب نسبة الحفاظ على الهيكل
        structure_preservation = len(new_lines) / max(len(original_lines), 1)
        structure_preservation = min(structure_preservation, 2.0) / 2.0  # تطبيع
        
        # تحليل التعقيد (نفضل المحتوى المعقد قليلاً)
        original_complexity = self.calculate_complexity(original_content)
        new_complexity = self.calculate_complexity(new_content)
        complexity_score = 0.5 + (new_complexity - original_complexity) * 0.5
        
        # الدرجة النهائية
        final_score = (structure_preservation * 0.6 + complexity_score * 0.4)
        return max(0.1, min(1.0, final_score))
    
    def calculate_complexity(self, text: str) -> float:
        """حساب تعقيد النص"""
        lines = text.split('\n')
        if not lines:
            return 0
        
        code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        avg_line_length = sum(len(line) for line in code_lines) / max(len(code_lines), 1)
        
        return min(avg_line_length / 100, 1.0)
    
    async def smart_merge(self, versions: List[Dict], original: str, instructions: str) -> str:
        """دمج ذكي للنسخ المتعددة"""
        if len(versions) == 1:
            return versions[0]["content"]
        
        # استخدام أفضل نموذج لدمج النسخ
        merger_model = "CHATGPT"  # يمكن تغييره لأي نموذج آخر
        
        merger_prompt = f"""أنت مساعد خبير في دمج الشيفرات البرمجية. لديك {len(versions)} نسخ مختلفة من نفس الملف، كل منها أنشئ بواسطة نموذج ذكاء اصطناعي مختلف.

التعليمات الأصلية: {instructions}

المطلوب:
1. قارن بين جميع النسخ
2. اختر أفضل الأجزاء من كل نسخة
3. ادمجها في نسخة واحدة متماسكة
4. تأكد من:
   - عدم وجود تعارضات بين الأجزاء
   - الحفاظ على التناسق في النمط
   - تطبيق التعليمات بدقة
   - صحة الشيفرة من الناحية الوظيفية

أعد النسخة المدمجة النهائية فقط:"""
        
        comparison_text = "\n\n".join([
            f"=== نسخة {v['model']} (جودة: {v['quality_score']:.2f}) ===\n{v['content']}" 
            for v in versions
        ])
        
        full_prompt = f"{merger_prompt}\n\n{comparison_text}"
        
        # استدعاء النموذج المدغم
        result = await self.call_standard_api(
            BACKEND_KEYS[merger_model], 
            "merged_version", 
            original, 
            full_prompt, 
            merger_model
        )
        
        if result.get("success"):
            return result["content"]
        else:
            # إذا فشل الدمج، نعود بأفضل نسخة
            best_version = max(versions, key=lambda x: x["quality_score"])
            return best_version["content"]
    
    async def call_standard_api(self, info: Dict, filename: str, content: str, instructions: str, model_name: str) -> Dict:
        """استدعاء API قياسي"""
        url = f"{info['base']}/chat/completions"
        
        system_prompt = f"""أنت مساعد برمجي خبير. الملف: {filename}

التعليمات: {instructions}

المطلوب:
1. تطبيق التعليمات بدقة عالية
2. الحفاظ على نمط البرمجة الأصلي  
3. إضافة تعليقات توضيحية عند الحاجة
4. التأكد من صحة الشيفرة

أعد الشيفرة المحسنة فقط:"""
        
        payload = {
            "model": info["models"][0],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"```\n{content}\n```"}
            ],
            "temperature": 0.1,
            "max_tokens": min(info["context"] - 1000, 32000)
        }
        
        headers = {
            "Authorization": f"Bearer {info['key']}",
            "Content-Type": "application/json"
        }
        
        try:
            async with asyncio.timeout(60):
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # استخراج الشيفرة من code blocks
                code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', content, re.DOTALL)
                if code_blocks:
                    content = code_blocks[0]
                else:
                    content = content.strip()
                
                return {
                    "success": True,
                    "content": content,
                    "tokens_used": result.get("usage", {}).get("total_tokens", 0)
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def call_openai_api(self, info: Dict, filename: str, content: str, instructions: str) -> Dict:
        """استدعاء OpenAI API"""
        return await self.call_standard_api(info, filename, content, instructions, "CHATGPT")
    
    async def call_claude_api(self, info: Dict, filename: str, content: str, instructions: str) -> Dict:
        """استدعاء Claude API"""
        url = f"{info['base']}/messages"
        
        system_prompt = f"""أنت مساعد برمجي خبير. الملف: {filename}

التعليمات: {instructions}

أعد الشيفرة المحسنة فقط:"""
        
        payload = {
            "model": info["models"][0],
            "max_tokens": 4000,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": f"```\n{content}\n```"
                }
            ]
        }
        
        headers = {
            "x-api-key": info["key"],
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        try:
            async with asyncio.timeout(60):
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                
                content = result.get("content", [{}])[0].get("text", "").strip()
                
                return {
                    "success": True,
                    "content": content,
                    "tokens_used": result.get("usage", {}).get("input_tokens", 0)
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def call_gemini_api(self, info: Dict, filename: str, content: str, instructions: str) -> Dict:
        """استدعاء Gemini API"""
        url = f"{info['base']}/models/{info['models'][0]}:generateContent?key={info['key']}"
        
        prompt = f"""أنت مساعد برمجي خبير. 

الملف: {filename}
التعليمات: {instructions}

المحتوى الأصلي:

أعد الشيفرة المحسنة فقط:"""
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 4000
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            async with asyncio.timeout(60):
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                
                content = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                
                return {
                    "success": True,
                    "content": content
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

# ================== واجهة المستخدم المحسنة ==================
async def start_enhanced(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """بدء محادثة مع نظام الدمج المتعدد"""
    user = update.effective_user
    
    # التحقق من النماذج المتاحة
    available_backends = validate_backend_keys()
    active_backends = [name for name, valid in available_backends.items() if valid]
    
    if not active_backends:
        await update.message.reply_text(
            "❌ **لا توجد نماذج متاحة**\n\nيرجى ضبط التوكنات أولاً.",
            parse_mode='Markdown'
        )
        return ConversationHandler.END
    
    user_id = user.id
    user_sessions[user_id] = {
        "selected_models": [],
        "merger": AdvancedMerger()
    }
    
    # إنشاء واجهة اختيار النماذج
    keyboard = []
    for backend in active_backends:
        emoji = "✅" if available_backends[backend] else "❌"
        keyboard.append([InlineKeyboardButton(f"{emoji} {backend}", callback_data=f"model_{backend}")])
    
    # إضافة خيارات الدمج
    keyboard.append([InlineKeyboardButton("🚀 جميع النماذج (دمج تلقائي)", callback_data="model_all")])
    keyboard.append([InlineKeyboardButton("🔀 دمج مخصص", callback_data="model_custom")])
    keyboard.append([InlineKeyboardButton("✅ تأكيد الاختيار", callback_data="model_confirm")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        f"🛠️ **مرحباً {user.first_name}!**\n\n"
        f"📊 **اختر نماذج الذكاء الاصطناعي:**\n"
        f"({len(active_backends)} نموذج متاح)\n\n"
        f"💡 **ميزة الدمج المتقدم:**\n"
        f"• إنشاء نسخ متعددة\n• مقارنة تلقائية\n• دمج ذكي\n• اختيار أفضل نسخة",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )
    
    return WAITING_BACKEND_CHOICE

async def handle_model_selection(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """معالجة اختيار النماذج"""
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data
    
    await query.answer()
    
    if user_id not in user_sessions:
        user_sessions[user_id] = {"selected_models": [], "merger": AdvancedMerger()}
    
    if data == "model_all":
        # اختيار جميع النماذج المتاحة
        available_backends = validate_backend_keys()
        user_sessions[user_id]["selected_models"] = [
            name for name, valid in available_backends.items() if valid
        ]
        
        await query.edit_message_text(
            f"✅ **تم اختيار جميع النماذج ({len(user_sessions[user_id]['selected_models'])})**\n\n"
            f"📁 **الآن أرسل الملف كـ Document**\n\n"
            f"🔧 سأقوم ب:\n"
            f"• إنشاء نسخة من كل نموذج\n• مقارنة النتائج\n• دمجها تلقائياً\n• إرجاع أفضل نسخة",
            parse_mode='Markdown'
        )
        return WAITING_FILE
    
    elif data == "model_custom":
        # اختيار مخصص
        await query.edit_message_text(
            "🔀 **الدمج المخصص**\n\n"
            "اكترب أسماء النماذج مفصولة بفاصلة:\n"
            "مثال: `CHATGPT, DEEPSEEK, CLAUDE`\n\n"
            "النماذج المتاحة: " + ", ".join(BACKEND_KEYS.keys()),
            parse_mode='Markdown'
        )
        return WAITING_CUSTOM_MODELS
    
    elif data == "model_confirm":
        if not user_sessions[user_id]["selected_models"]:
            await query.edit_message_text(
                "❌ لم تختر أي نماذج. اختر نماذج أولاً.",
                parse_mode='Markdown'
            )
            return WAITING_BACKEND_CHOICE
        
        await query.edit_message_text(
            f"✅ **تم اختيار {len(user_sessions[user_id]['selected_models'])} نماذج**\n\n"
            f"📁 **الآن أرسل الملف كـ Document**\n\n"
            f"النماذج المختارة: {', '.join(user_sessions[user_id]['selected_models'])}",
            parse_mode='Markdown'
        )
        return WAITING_FILE
    
    else:
        # اختيار نموذج فردي
        model_name = data.replace("model_", "")
        if model_name in user_sessions[user_id]["selected_models"]:
            user_sessions[user_id]["selected_models"].remove(model_name)
            await query.answer(f"تم إزالة {model_name}")
        else:
            user_sessions[user_id]["selected_models"].append(model_name)
            await query.answer(f"تم إضافة {model_name}")
        
        # تحديث الواجهة
        available_backends = validate_backend_keys()
        keyboard = []
        for backend in available_backends:
            emoji = "🟢" if backend in user_sessions[user_id]["selected_models"] else "⚪"
            status = "✅" if available_backends[backend] else "❌"
            keyboard.append([InlineKeyboardButton(f"{emoji} {status} {backend}", callback_data=f"model_{backend}")])
        
        keyboard.append([InlineKeyboardButton("🚀 جميع النماذج", callback_data="model_all")])
        keyboard.append([InlineKeyboardButton("🔀 دمج مخصص", callback_data="model_custom")])
        keyboard.append([InlineKeyboardButton(f"✅ تأكيد ({len(user_sessions[user_id]['selected_models'])})", callback_data="model_confirm")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            f"🔧 **اختر النماذج ({len(user_sessions[user_id]['selected_models'])} مختارة)**\n\n"
            f"المختارة: {', '.join(user_sessions[user_id]['selected_models']) or 'لا شيء'}",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        return WAITING_BACKEND_CHOICE

async def handle_custom_models(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """معالجة الاختيار المخصص للنماذج"""
    user_id = update.effective_user.id
    text = update.message.text
    
    if user_id not in user_sessions:
        user_sessions[user_id] = {"selected_models": [], "merger": AdvancedMerger()}
    
    # تحليل النماذج المدخلة
    input_models = [model.strip().upper() for model in text.split(',')]
    valid_models = []
    
    for model in input_models:
        if model in BACKEND_KEYS:
            valid_models.append(model)
        else:
            await update.message.reply_text(f"❌ النموذج {model} غير معروف")
    
    if not valid_models:
        await update.message.reply_text("❌ لم تدخل أي نماذج صحيحة")
        return WAITING_BACKEND_CHOICE
    
    user_sessions[user_id]["selected_models"] = valid_models
    
    await update.message.reply_text(
        f"✅ **تم اختيار {len(valid_models)} نماذج**\n\n"
        f"📁 **الآن أرسل الملف كـ Document**\n\n"
        f"النماذج: {', '.join(valid_models)}",
        parse_mode='Markdown'
    )
    return WAITING_FILE

# ================== معالجة الملف مع الدمج المتقدم ==================
async def process_file_with_merging(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """معالجة الملف مع نظام الدمج المتعدد"""
    user_id = update.effective_user.id
    
    if user_id not in user_sessions or not user_sessions[user_id].get("selected_models"):
        await update.message.reply_text("❌ لم تختر أي نماذج. ابدأ بـ /start")
        return ConversationHandler.END
    
    if not update.message.document:
        await update.message.reply_text("❌ يرجى إرسال ملف")
        return WAITING_FILE
    
    # تحميل الملف
    file = await ctx.bot.get_file(update.message.document.file_id)
    filename = update.message.document.file_name or "unknown"
    
    progress_msg = await update.message.reply_text("📥 **جاري تحميل الملف...**")
    
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, sanitize_filename(filename))
            await file.download_to_drive(custom_path=path)
            
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        
        await progress_msg.edit_text("🔄 **جاري إنشاء النسخ المتعددة...**")
        
        # إنشاء نسخ متعددة باستخدام النماذج المختارة
        merger = user_sessions[user_id]["merger"]
        models = user_sessions[user_id]["selected_models"]
        
        versions = await merger.create_multiple_versions(
            filename, content, "تحسين الشيفرة", models
        )
        
        if not versions:
            await progress_msg.edit_text("❌ فشلت جميع النماذج في معالجة الملف")
            return ConversationHandler.END
        
        await progress_msg.edit_text(f"🔀 **جاري دمج {len(versions)} نسخة...**")
        
        # دمج النسخ
        merge_result = await merger.merge_versions(versions, content, "تحسين الشيفرة")
        
        if merge_result["success"]:
            # حفظ النتيجة
            backup_path = create_backup(user_id, f"merged_{filename}", merge_result["content"])
            
            # إرسال النتيجة
            result_text = f"""
🎉 **تم المعالجة بنجاح!**

📊 **النتيجة:**
• الملف: `{filename}`
• النماذج المستخدمة: {len(versions)}
• طريقة الدمج: {merge_result['method']}
• جودة الدمج: {merge_result.get('compatibility_score', 'N/A')}

💾 **النسخة المحسنة:**"""
            
            await progress_msg.edit_text(result_text, parse_mode='Markdown')
            
            # إرسال الملف المحسن
            with tempfile.NamedTemporaryFile(mode='w', suffix=filename, encoding='utf-8', delete=False) as f:
                f.write(merge_result["content"])
                temp_path = f.name
            
            try:
                with open(temp_path, 'rb') as file_obj:
                    await update.message.reply_document(
                        document=file_obj,
                        filename=f"enhanced_{filename}",
                        caption=f"🛠️ {filename} - النسخة المدمجة ({len(versions)} نموذج)"
                    )
            finally:
                os.unlink(temp_path)
            
            # إرسال تفاصيل إضافية
            details_text = f"""
📈 **تفاصيل المعالجة:**
"""
            for version in versions:
                details_text += f"• {version['model']}: جودة {version['quality_score']:.2f}\n"
            
            await update.message.reply_text(details_text, parse_mode='Markdown')
            
        else:
            await progress_msg.edit_text(f"❌ فشل الدمج: {merge_result.get('error', 'خطأ غير معروف')}")
        
    except Exception as e:
        logging.error(f"Processing error: {e}")
        await progress_msg.edit_text(f"❌ خطأ في المعالجة: {str(e)}")
    
    return ConversationHandler.END

# ================== دوال مساعدة موجودة سابقاً ==================
def validate_token(token: str) -> bool:
    pattern = r'^\d+:[A-Za-z0-9_-]+$'
    return bool(re.match(pattern, token))

def sanitize_filename(filename: str) -> str:
    return re.sub(r'[^\w\-_.]', '_', filename)

def detect_language(filename: str) -> str:
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
    validation_results = {}
    
    for name, config in BACKEND_KEYS.items():
        if config["key"].startswith("sk-your-"):
            validation_results[name] = False
        else:
            validation_results[name] = True
    
    return validation_results

# ================== التشغيل الرئيسي ==================
def main():
    """تشغيل البوت مع نظام الدمج المتقدم"""
    
    # إنشاء الهيكل التنظيمي
    os.makedirs("backups", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # توكن البوت
    BOT_TOKEN = "8214334664:AAGWEhTYrFTyN_TCxbFlQfdnIKYLgfI496A"
    
    # التحقق من النماذج المتاحة
    available_models = validate_backend_keys()
    active_models = [name for name, valid in available_models.items() if valid]
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/advanced_bot.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    if not active_models:
        logger.warning("⚠️  لا توجد نماذج مضبوطة!")
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
        entry_points=[CommandHandler("start", start_enhanced)],
        states={
            WAITING_BACKEND_CHOICE: [CallbackQueryHandler(handle_model_selection, pattern="^model_")],
            WAITING_CUSTOM_MODELS: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_custom_models)],
            WAITING_FILE: [MessageHandler(filters.Document.ALL, process_file_with_merging)]
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        name="advanced_conv",
        allow_reentry=True
    )

    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("setup", setup_command))
    app.add_handler(CommandHandler("help", help_enhanced))

    logger.info("🚀 بدء تشغيل البوت المتقدم مع نظام الدمج...")
    
    try:
        app.run_polling()
    except Exception as e:
        logger.error(f"❌ فشل تشغيل البوت: {e}")
        raise

# إضافة الدوال المساعدة المفقودة
async def cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("تم الإلغاء")
    return ConversationHandler.END

async def setup_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    validation = validate_backend_keys()
    setup_text = "⚙️ **إعدادات النماذج:**\n\n"
    
    for name, config in BACKEND_KEYS.items():
        status = "✅ جاهز" if validation[name] else "❌ يحتاج إعداد"
        models = ", ".join(config["models"])
        setup_text += f"**{name}:** {status}\n"
        setup_text += f"   • النماذج: {models}\n\n"
    
    await update.message.reply_text(setup_text, parse_mode='Markdown')

async def help_enhanced(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    help_text = """
🛠️ **بوت الدمج المتقدم - التعليمات**

🎯 **الميزات الجديدة:**
• دعم 7+ نماذج ذكاء اصطناعي
• نظام دمج ذكي للنسخ المتعددة
• مقارنة تلقائية بين النتائج
• اختيار أفضل نسخة تلقائياً

📝 **طريقة الاستخدام:**
1. /start - بدء جلسة جديدة
2. اختر النماذج المطلوبة
3. أرسل الملف كـ Document
4. استلم النسخة المدمجة المحسنة

🔧 **النماذج المدعومة:**
ChatGPT, Claude, Gemini, DeepSeek, Kimi, Qianwen, Zhipu
    """
    await update.message.reply_text(help_text, parse_mode='Markdown')

if __name__ == "__main__":
    main()
