
import asyncio
import logging
import sys
import os
import sqlite3
import pandas as pd
import nest_asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatType
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message, FSInputFile
from cachetools import TTLCache

nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# ==========================================
# НАСТРОЙКИ
# ==========================================
BOT_TOKEN = ""
MY_TELEGRAM_ID = 

CSV_FILE_NAME = "Answers_hackaton_final.csv"
DB_FILE_NAME = "moderation_logs.db"
DB_TABLE_NAME = "logs"

# Проверка наличия функций нейросети в памяти
try:
    predict_func = predict_two_stage
    init_func = initialize_runtime
    TRASH_LBL = TRASH_LABEL_NAME
except NameError:
    logger.error("КРИТИЧЕСКАЯ ОШИБКА: Сначала запустите ячейку с кодом нейросети.")
    sys.exit(1)


# ==========================================
# БАЗА ДАННЫХ SQLITE3
# ==========================================
def init_db(db_path: str = DB_FILE_NAME) -> None:
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {DB_TABLE_NAME} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Date DATETIME NOT NULL,
        User_ID INTEGER NOT NULL,
        Text TEXT NOT NULL,
        Is_Junk BOOLEAN NOT NULL,
        Intent TEXT,
        Confidence REAL NOT NULL DEFAULT 0.0,
        Status TEXT NOT NULL,
        Final_Decision TEXT,
        Is_Text_Changed BOOLEAN NOT NULL DEFAULT 0
    );
    """
    with sqlite3.connect(db_path) as connection:
        connection.execute(create_table_sql)
        connection.commit()

def db_insert_log(date_value: str, user_id: int, text: str, is_junk: bool, intent: str, confidence: float, status: str, final_decision: Optional[str], is_text_changed: bool, db_path: str = DB_FILE_NAME) -> int:
    insert_sql = f"INSERT INTO {DB_TABLE_NAME} (Date, User_ID, Text, Is_Junk, Intent, Confidence, Status, Final_Decision, Is_Text_Changed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);"
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        cursor.execute(insert_sql, (date_value, user_id, text, int(is_junk), intent, float(confidence), status, final_decision, int(is_text_changed)))
        connection.commit()
        return int(cursor.lastrowid)

def db_update_log(log_id: int, db_path: str = DB_FILE_NAME, **fields: object) -> None:
    if not fields: return
    normalized_fields = dict(fields)
    if "Is_Junk" in normalized_fields: normalized_fields["Is_Junk"] = int(bool(normalized_fields["Is_Junk"]))
    if "Is_Text_Changed" in normalized_fields: normalized_fields["Is_Text_Changed"] = int(bool(normalized_fields["Is_Text_Changed"]))

    assignments = ", ".join(f"{column} = ?" for column in normalized_fields.keys())
    values = list(normalized_fields.values())
    values.append(log_id)

    update_sql = f"UPDATE {DB_TABLE_NAME} SET {assignments} WHERE id = ?;"
    with sqlite3.connect(db_path) as connection:
        connection.execute(update_sql, values)
        connection.commit()


# ==========================================
# АНАЛИТИКА И ОТЧЕТНОСТЬ (MarkdownV2)
# ==========================================
def escape_markdown_v2(text: str) -> str:
    markdown_v2_special_chars = set("\\_*[]()~`>#+-=|{}.!")
    return "".join(f"\\{char}" if char in markdown_v2_special_chars else char for char in text)

def generate_analytics_report(db_path: str) -> str:
    def percent(part: int, total: int) -> float:
        return (part / total * 100.0) if total > 0 else 0.0

    escape = escape_markdown_v2

    try:
        with sqlite3.connect(db_path) as connection:
            cursor = connection.cursor()

            cursor.execute(f"SELECT COUNT(*) FROM {DB_TABLE_NAME};")
            total_messages = int(cursor.fetchone()[0] or 0)

            cursor.execute(f"SELECT COUNT(*) FROM {DB_TABLE_NAME} WHERE Is_Junk = 1;")
            junk_messages = int(cursor.fetchone()[0] or 0)

            cursor.execute(f"SELECT COUNT(*) FROM {DB_TABLE_NAME} WHERE Is_Junk = 0;")
            useful_messages = int(cursor.fetchone()[0] or 0)

            cursor.execute(f"SELECT Intent, COUNT(*) AS intent_count FROM {DB_TABLE_NAME} WHERE Is_Junk = 0 AND Intent IS NOT NULL AND TRIM(Intent) <> '' GROUP BY Intent ORDER BY intent_count DESC, Intent ASC LIMIT 3;")
            top_intents = cursor.fetchall()

            cursor.execute(f"SELECT COUNT(*) FROM {DB_TABLE_NAME} WHERE Status = 'Approved' AND Is_Text_Changed = 0;")
            auto_approved_count = int(cursor.fetchone()[0] or 0)

            cursor.execute(f"SELECT COUNT(*) FROM {DB_TABLE_NAME} WHERE Status = 'Approved' AND Is_Text_Changed = 1;")
            manual_approved_count = int(cursor.fetchone()[0] or 0)

            cursor.execute(f"SELECT AVG(Confidence) FROM {DB_TABLE_NAME};")
            avg_confidence_raw = cursor.fetchone()[0]
            avg_confidence = float(avg_confidence_raw) if avg_confidence_raw is not None else 0.0

            cursor.execute(f"SELECT COUNT(*) FROM {DB_TABLE_NAME} WHERE Confidence < 0.60;")
            low_confidence_count = int(cursor.fetchone()[0] or 0)

            cursor.execute(f"SELECT COUNT(*) FROM {DB_TABLE_NAME} WHERE Status = 'Pending';")
            pending_count = int(cursor.fetchone()[0] or 0)

            cursor.execute(f"SELECT COUNT(*) FROM {DB_TABLE_NAME} WHERE Status IN ('Approved', 'Trashed');")
            processed_count = int(cursor.fetchone()[0] or 0)

    except sqlite3.Error as exc:
        logger.error(f"Ошибка формирования отчета: {exc}")
        return f"*{escape('Аналитический отчет')}*\n\n{escape('Не удалось сформировать данные. Проверьте базу.')}"

    lines = [
        f"*{escape('Аналитический отчет (Executive Summary)')}*",
        "",
        f"*{escape('1. Воронка фильтрации')}*",
        f"• {escape(f'Всего входящих сообщений: {total_messages}')}",
        f"• {escape(f'Отфильтровано как мусор: {junk_messages} ({percent(junk_messages, total_messages):.2f}%)')}",
        f"• {escape(f'Полезные обращения: {useful_messages} ({percent(useful_messages, total_messages):.2f}%)')}",
        "",
        f"*{escape('2. Топ проблем (Интенты)')}*",
    ]

    if top_intents:
        for index, (intent, intent_count) in enumerate(top_intents, start=1):
            count_value = int(intent_count)
            intent_share = percent(count_value, useful_messages)
            lines.append(f"• {escape(f'{index}) {str(intent)} — {count_value} ({intent_share:.2f}%)')}")
    else:
        lines.append(f"• {escape('Данные по интентам отсутствуют.')}")

    lines.extend([
        "",
        f"*{escape('3. Индекс автоматизации (Human-in-the-Loop)')}*",
        f"• {escape(f'Ответы по готовому шаблону: {auto_approved_count} ({percent(auto_approved_count, total_messages):.2f}%)')}",
        f"• {escape(f'Ответы с ручной корректировкой: {manual_approved_count} ({percent(manual_approved_count, total_messages):.2f}%)')}",
        "",
        f"*{escape('4. Метрики качества нейросети (AI Health)')}*",
        f"• {escape(f'Средняя уверенность модели: {avg_confidence * 100:.2f}%')}",
        f"• {escape(f'Неуверенные предсказания (< 60%): {low_confidence_count} ({percent(low_confidence_count, total_messages):.2f}%)')}",
        "",
        f"*{escape('5. Статус очереди (SLA)')}*",
        f"• {escape(f'Ожидают проверки: {pending_count}')}",
        f"• {escape(f'Обработано модератором: {processed_count}')}",
    ])

    return "\n".join(lines)


# ==========================================
# ЗАГРУЗКА БАЗЫ ИЗ CSV ДЛЯ РУЧНОЙ СМЕНЫ ИНТЕНТА
# ==========================================
ANSWERS_BASE: Dict[str, str] = {}

def load_answers():
    global ANSWERS_BASE
    try:
        df = pd.read_csv(CSV_FILE_NAME, sep=None, engine='python')
        i_col = next((c for c in df.columns if c.lower() in ['intent', 'интент', 'label']), None)
        a_col = next((c for c in df.columns if c.lower() in ['ответ', 'answer', 'reply', 'text_answer']), None)
        if i_col and a_col:
            for _, row in df.dropna(subset=[i_col, a_col]).iterrows():
                ANSWERS_BASE[str(row[i_col]).strip()] = str(row[a_col]).strip()
            logger.info(f"База данных успешно загружена: {len(ANSWERS_BASE)} шаблонов ответов.")
        else:
            logger.error("Колонки 'Intent' или 'Ответ' не найдены в CSV-файле.")
    except Exception as e:
        logger.error(f"Ошибка при обработке CSV: {e}")


# ==========================================
# FSM, КНОПКИ И СТРУКТУРЫ ДАННЫХ
# ==========================================
class ModStates(StatesGroup):
    editing_reply = State()

class ModCb(CallbackData, prefix="m"):
    a: str
    id: int

class TopicCb(CallbackData, prefix="t"):
    id: int
    key: str

@dataclass
class PendingMsg:
    db_id: int
    group_id: int
    msg_id: int
    user_text: str
    intent: str
    reply: str
    is_junk: bool
    is_text_changed: bool = False

moderation_queue: TTLCache = TTLCache(maxsize=1000, ttl=86400)
router = Router()

def get_card_text(rec: PendingMsg) -> str:
    return (
        f"МОДЕРАЦИЯ СООБЩЕНИЯ\n\n"
        f"Вопрос пользователя: {rec.user_text}\n"
        f"Определенный интент: {rec.intent}\n\n"
        f"Текст ответа для отправки:\n{rec.reply}"
    )

def get_main_kb(rid: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="ОДОБРИТЬ И ОТПРАВИТЬ", callback_data=ModCb(a="ok", id=rid).pack())],
        [
            InlineKeyboardButton(text="ИЗМЕНИТЬ ОТВЕТ", callback_data=ModCb(a="edit", id=rid).pack()),
            InlineKeyboardButton(text="СМЕНИТЬ ИНТЕНТ", callback_data=ModCb(a="topic", id=rid).pack())
        ],
        [InlineKeyboardButton(text="В МУСОР", callback_data=ModCb(a="trash", id=rid).pack())]
    ])

def get_topics_kb(rid: int) -> InlineKeyboardMarkup:
    buttons = []
    for intent in ANSWERS_BASE.keys():
        buttons.append([InlineKeyboardButton(text=intent[:30], callback_data=TopicCb(id=rid, key=intent[:20]).pack())])
    buttons.append([InlineKeyboardButton(text="НАЗАД", callback_data=ModCb(a="back", id=rid).pack())])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


# ==========================================
# ОБРАБОТЧИК /db
# ==========================================
@router.message(Command("db"))
async def send_db_dump(message: Message) -> None:
    if message.from_user and message.from_user.id != MY_TELEGRAM_ID:
        return
    if not os.path.exists(DB_FILE_NAME):
        await message.answer("Файл базы данных не найден.")
        return

    analytics_report = generate_analytics_report(DB_FILE_NAME)
    await message.answer(analytics_report, parse_mode="MarkdownV2")

    document = FSInputFile(DB_FILE_NAME)
    await message.answer_document(document=document, caption="Выгрузка базы данных moderation_logs.db")


# ==========================================
# ОБРАБОТЧИК ВХОДЯЩИХ СООБЩЕНИЙ ИЗ ГРУППЫ
# ==========================================
@router.message(F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}))
async def handle_group_msg(message: Message, bot: Bot):
    if message.from_user and message.from_user.id == bot.id: return
    text = message.text or message.caption or ""
    if not text.strip(): return

    pred = await asyncio.to_thread(predict_func, text)
    intent = str(pred.get("label", ""))
    confidence = float(pred.get("confidence", 0.0))
    is_junk = (intent == TRASH_LBL)
    reply = str(pred.get("answer", "Шаблон ответа не найден."))

    user_id = message.from_user.id if message.from_user else 0
    created_at = (message.date.replace(tzinfo=None) if message.date else datetime.utcnow()).strftime("%Y-%m-%d %H:%M:%S")

    try:
        db_id = db_insert_log(
            date_value=created_at, user_id=user_id, text=text, is_junk=is_junk,
            intent=intent, confidence=confidence, status="Pending",
            final_decision=None, is_text_changed=False
        )
    except Exception as exc:
        logger.error(f"Ошибка записи в БД: {exc}")
        return

    rec = PendingMsg(
        db_id=db_id,
        group_id=message.chat.id,
        msg_id=message.message_id,
        user_text=text,
        intent=intent,
        reply=reply,
        is_junk=is_junk
    )

    try:
        card = await bot.send_message(chat_id=MY_TELEGRAM_ID, text=get_card_text(rec), reply_markup=get_main_kb(0))
        moderation_queue[card.message_id] = rec
        await bot.edit_message_reply_markup(chat_id=MY_TELEGRAM_ID, message_id=card.message_id, reply_markup=get_main_kb(card.message_id))
    except Exception as e:
        logger.error(f"Ошибка отправки карточки модерации: {e}")


# ==========================================
# ОБРАБОТЧИКИ КНОПОК
# ==========================================
@router.callback_query(ModCb.filter(F.a == "ok"))
async def action_ok(call: CallbackQuery, callback_data: ModCb, bot: Bot):
    rec = moderation_queue.get(callback_data.id)
    if not rec: return await call.answer("Запрос устарел или не найден.")
    try:
        await bot.send_message(chat_id=rec.group_id, text=rec.reply, reply_to_message_id=rec.msg_id)
        await call.message.edit_text(f"ОТПРАВЛЕНО ПОЛЬЗОВАТЕЛЮ:\n\n{rec.reply}")

        # Записываем финальный интент в базу
        db_update_log(rec.db_id, Status="Approved", Final_Decision=rec.intent, Intent=rec.intent, Is_Text_Changed=rec.is_text_changed)
    except Exception as e:
        await call.answer(f"Ошибка при отправке: {e}", show_alert=True)
    await call.answer()

@router.callback_query(ModCb.filter(F.a == "trash"))
async def action_trash(call: CallbackQuery, callback_data: ModCb):
    rec = moderation_queue.get(callback_data.id)
    if rec:
        # Записываем "Мусор" как финальный интент
        db_update_log(rec.db_id, Status="Trashed", Final_Decision=TRASH_LBL, Is_Junk=True, Intent=TRASH_LBL, Is_Text_Changed=rec.is_text_changed)
    await call.message.edit_text("Сообщение перемещено в мусор.")
    await call.answer()

@router.callback_query(ModCb.filter(F.a == "topic"))
async def action_show_topics(call: CallbackQuery, callback_data: ModCb):
    await call.message.edit_reply_markup(reply_markup=get_topics_kb(callback_data.id))
    await call.answer()

@router.callback_query(ModCb.filter(F.a == "back"))
async def action_back(call: CallbackQuery, callback_data: ModCb):
    await call.message.edit_reply_markup(reply_markup=get_main_kb(callback_data.id))
    await call.answer()

@router.callback_query(TopicCb.filter())
async def action_topic_selected(call: CallbackQuery, callback_data: TopicCb, bot: Bot):
    rec = moderation_queue.get(callback_data.id)
    if not rec: return

    full_intent = next((k for k in ANSWERS_BASE.keys() if k[:20] == callback_data.key), None)
    if full_intent:
        rec.intent = full_intent
        rec.reply = ANSWERS_BASE.get(full_intent, "Шаблон отсутствует.")
        rec.is_text_changed = True

        db_update_log(rec.db_id, Intent=rec.intent, Is_Text_Changed=True)
        await bot.edit_message_text(chat_id=MY_TELEGRAM_ID, message_id=callback_data.id, text=get_card_text(rec), reply_markup=get_main_kb(callback_data.id))
    await call.answer("Категория и текст ответа обновлены.")

@router.callback_query(ModCb.filter(F.a == "edit"))
async def action_edit_reply(call: CallbackQuery, callback_data: ModCb, state: FSMContext):
    await state.set_state(ModStates.editing_reply)
    await state.update_data(rid=callback_data.id)
    await call.message.reply("Введите новый текст ответа (он заменит текущий шаблон):")
    await call.answer()

@router.message(ModStates.editing_reply)
async def process_new_reply(message: Message, state: FSMContext, bot: Bot):
    if message.chat.type != ChatType.PRIVATE: return
    data = await state.get_data()
    rid = data.get("rid")
    rec = moderation_queue.get(rid)
    if rec and message.text:
        rec.reply = message.text.strip()
        rec.is_text_changed = True

        db_update_log(rec.db_id, Is_Text_Changed=True)
        await bot.edit_message_text(chat_id=MY_TELEGRAM_ID, message_id=rid, text=get_card_text(rec), reply_markup=get_main_kb(rid))
        await message.answer("Текст ответа обновлен. Нажмите 'ОДОБРИТЬ И ОТПРАВИТЬ' для завершения.")
    await state.clear()

# ==========================================
# ИНИЦИАЛИЗАЦИЯ И ЗАПУСК
# ==========================================
async def main():
    logger.info("Запуск инициализации конвейера ML и базы данных...")
    init_func()
    load_answers()
    init_db()

    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    logger.info("Инициализация завершена. Бот запущен и ожидает сообщения.")
    logger.info("Для выгрузки логов базы данных отправьте боту команду /db")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())