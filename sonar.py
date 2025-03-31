import logging
import httpx
import time
import asyncio
import re
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.constants import ChatAction

from telegram.ext import Application, MessageHandler, CallbackContext, filters, CallbackQueryHandler
import telegramify_markdown

TELEGRAM_TOKEN = 'BOT_TOKEN'
PERPLEXITY_API_KEY = 'PERPLEXITY_KEY'
PERPLEXITY_API_BASE = 'https://api.perplexity.ai'

MODEL_SONAR = "sonar" 
MODEL_DEEP_RESEARCH = "sonar-deep-research" 

DEFAULT_MODEL = MODEL_SONAR
SYSTEM_PROMPT = """instructions here"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

chat_states = {}

async_http_client = httpx.AsyncClient(timeout=120.0) 

def split_text(text, max_length=4050):
    parts = []
    current_chunk = ""
    lines = text.split('\n')

    for line in lines:

        if len(current_chunk) + len(line) + 1 > max_length:

            if current_chunk:
                parts.append(current_chunk)

            if len(line) > max_length:
                for i in range(0, len(line), max_length):
                    parts.append(line[i:i+max_length])
                current_chunk = "" 
            else:

                current_chunk = line
        else:

            if current_chunk:
                current_chunk += "\n" + line
            else:
                current_chunk = line

    if current_chunk:
        parts.append(current_chunk)

    final_parts = []
    for part in parts:
        if len(part) > max_length:
             for i in range(0, len(part), max_length):
                final_parts.append(part[i:i+max_length])
        else:
            final_parts.append(part)

    return final_parts

def remove_think_tags(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

async def query_perplexity(message_content: str, conversation: list, model: str):
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    if not conversation:
        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    conversation.append({"role": "user", "content": message_content})

    payload = {
        "model": model,
        "messages": conversation,
        "temperature": 0.5,
    }

    try:
        logger.info(f"Sending request to Perplexity API. Model: {model}, Payload Messages Count: {len(conversation)}")
        resp = await async_http_client.post(
            f"{PERPLEXITY_API_BASE}/chat/completions",
            headers=headers,
            json=payload
        )
        resp.raise_for_status()

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Error calling Perplexity API: {e.response.status_code} - {e.response.text}")
        error_message = f"Error: API returned status {e.response.status_code}."
        try:
            error_detail = e.response.json().get("detail", {}).get("message", e.response.text)
            error_message += f"\nDetails: {error_detail}"
        except Exception:
            error_message += f"\nRaw Response: {e.response.text}"
        return error_message, conversation[:-1], None
    except httpx.RequestError as e:
        logger.error(f"Request Error calling Perplexity API: {e}")
        return f"Error: Could not connect to Perplexity API. ({type(e).__name__})", conversation[:-1], None
    except Exception as e:
        logger.error(f"Unexpected error during Perplexity API call: {e}", exc_info=True)
        return f"Error: An unexpected error occurred during the API call. ({type(e).__name__})", conversation[:-1], None

    try:
        result = resp.json()
        logger.info(f"Raw Perplexity response structure (top-level keys): {list(result.keys())}")

        if not result or "choices" not in result or not result["choices"]:
             logger.error(f"Invalid response structure from Perplexity: {result}")
             return "Error: Received an invalid or empty response from the API.", conversation, None

        assistant_message = result["choices"][0].get("message", {})
        reply_text = assistant_message.get("content")

        if reply_text is None:
            logger.error(f"No 'content' found in assistant message: {assistant_message}")
            return "Error: API response did not contain message content.", conversation, None

        logger.info(f"Assistant reply content received: {reply_text[:200]}...")
        reply_text = remove_think_tags(reply_text)
        conversation.append({"role": "assistant", "content": reply_text})

        citations_list = None
        citations_data = result.get("citations")
        logger.info(f"Raw citations data found at top-level 'citations': {citations_data}")

        if citations_data:
            if isinstance(citations_data, list):
                try:
                    if citations_data:

                        logger.info(f"Structure of first citation element: {citations_data[0]}")

                    citations_list = [
                        f"{idx+1}. {cite}" 
                        for idx, cite in enumerate(citations_data)
                        if isinstance(cite, str) and cite 
                    ]
                    logger.info(f"Successfully formatted {len(citations_list)} citations from list of strings.")

                except Exception as format_err:
                    logger.error(f"Error formatting citations list (strings): {format_err}", exc_info=True)
                    logger.error(f"Problematic citations_data during formatting: {citations_data}")
                    citations_list = None
            else:
                logger.warning(f"Citations data received ('citations') but is not a list: Type={type(citations_data)}, Data={citations_data}")
                citations_list = None
        else:
            logger.info("No citations data found in top-level 'citations'.")
            citations_list = None

        return reply_text, conversation, citations_list

    except Exception as e:
        logger.error(f"Error parsing Perplexity response: {e}", exc_info=True)
        logger.error(f"Problematic API Response Text during parsing: {resp.text}")
        return "Error: Could not parse the API response.", conversation, None

async def handle_message(update: Update, context: CallbackContext):
    if not update.message or not update.message.text:
        return

    chat_id = update.message.chat.id
    message_id = update.message.message_id
    user_content = update.message.text

    logger.info(f"Received message from chat_id: {chat_id} (message_id: {message_id}): {user_content[:50]}...")

    chat_state = chat_states.get(chat_id, {"history": [], "model": DEFAULT_MODEL})
    conversation = chat_state["history"]
    current_model = chat_state["model"]
    logger.info(f"Using model '{current_model}' for chat {chat_id}")

    typing_task = None
    stop_typing = asyncio.Event()
    async def keep_typing():
        while not stop_typing.is_set():
            try:
                await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
                logger.debug(f"Sent typing action to {chat_id}")
            except Exception as e:
                logger.warning(f"Error sending chat action to {chat_id}: {e}")
            await asyncio.sleep(4) 

    typing_task = asyncio.create_task(keep_typing())

    reply_text, updated_conversation, citations = None, conversation, None
    try:
        reply_text, updated_conversation, citations = await query_perplexity(
            user_content, conversation.copy(), current_model
        )
    except Exception as e:
        logger.error(f"Unexpected error in handle_message during API call: {e}", exc_info=True)
        reply_text = f"Error: An unexpected error occurred processing your request."
        updated_conversation = conversation
        citations = None 
    finally:
        if typing_task:
            stop_typing.set()
            try:
                await asyncio.wait_for(typing_task, timeout=1)
            except asyncio.TimeoutError:
                logger.debug("Typing task didn't finish immediately.")
            except Exception as e:
                 logger.warning(f"Error waiting for typing task: {e}")

    chat_states[chat_id] = {"history": updated_conversation, "model": current_model}
    logger.info(f"Conversation history length for chat {chat_id}: {len(updated_conversation)}")

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton(text="✨ New Sonar", callback_data="new_sonar"),
         InlineKeyboardButton(text="✨ New Deep Research", callback_data="new_deep_research")]
    ])

    if reply_text.startswith("Error:"):

        reply_text += ("\n\nTap the buttons on the last successful message or below (if citations are shown) to start a new conversation.")

        citations = None

    try:

        try:
            converted_reply = telegramify_markdown.markdownify(reply_text)
        except Exception as md_err:
            logger.warning(f"Markdown conversion failed for main reply: {md_err}. Sending plain text.")
            converted_reply = telegramify_markdown.escape_markdown(reply_text)

        logger.info(f"Sending response to chat_id: {chat_id} (length: {len(converted_reply)})")
        chunks = split_text(converted_reply)
        total_chunks = len(chunks)

        for idx, chunk in enumerate(chunks):

            reply_markup = None
            is_last_chunk = (idx == total_chunks - 1)

            if is_last_chunk and not citations:
                reply_markup = keyboard
                logger.info(f"Adding keyboard to last chunk (chunk {idx+1}) as there are no citations.")

            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode="MarkdownV2",
                    reply_markup=reply_markup, 
                    disable_web_page_preview=True
                )
                logger.debug(f"Sent chunk {idx+1}/{total_chunks} to chat_id: {chat_id}")
            except Exception as send_err:
                logger.error(f"Error sending chunk {idx+1}/{total_chunks} to {chat_id}: {send_err}", exc_info=True)

                if "can't parse entities" in str(send_err).lower():
                    try:
                        logger.warning(f"Retrying chunk {idx+1} as plain text.")
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=chunk,
                            reply_markup=reply_markup, 
                            disable_web_page_preview=True
                        )
                    except Exception as retry_err:
                        logger.error(f"Retry sending chunk {idx+1} as plain text also failed: {retry_err}")

        if citations:
            citations_text = "\n".join(citations)
            citations_message = f"*Citations:*\n{citations_text}"
            try:

                try:
                    formatted_citations = telegramify_markdown.markdownify(citations_message)
                except Exception as md_cite_err:
                    logger.warning(f"Markdown conversion failed for citations: {md_cite_err}. Sending plain.")
                    formatted_citations = telegramify_markdown.escape_markdown(citations_message)

                await context.bot.send_message(
                    chat_id=chat_id,
                    text=formatted_citations,
                    parse_mode="MarkdownV2",
                    disable_web_page_preview=True,
                    reply_markup=keyboard 
                )
                logger.info(f"Sent citations block with keyboard to chat_id: {chat_id}")
            except Exception as e:
                logger.error(f"Error sending citations block to {chat_id}: {e}", exc_info=True)
        else:

             if not reply_text.startswith("Error:"): 
                 logger.info(f"No citations to send for chat_id: {chat_id}. Keyboard attached to last main chunk.")

    except Exception as e:

        logger.error(f"General error formatting/sending response for chat {chat_id}: {e}", exc_info=True)
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Sorry, an error occurred while preparing the response."
            )
        except Exception as final_err:
            logger.error(f"Failed to send even the generic error message to {chat_id}: {final_err}")

async def _start_new_conversation(chat_id: int, context: CallbackContext, model_id: str, model_name: str):
    global chat_states
    chat_states[chat_id] = {"history": [], "model": model_id}
    logger.info(f"Started new conversation for chat {chat_id}. Model set to: {model_id} ({model_name})")
    message = f"✨ New {model_name} conversation started. How can I help you?"
    try:
        await context.bot.send_message(
            chat_id=chat_id,
            text=telegramify_markdown.markdownify(message),
            parse_mode="MarkdownV2"
        )
    except Exception as e:
        logger.error(f"Error sending new conversation message to {chat_id}: {e}")

async def new_sonar_callback(update: Update, context: CallbackContext):
    query = update.callback_query
    if not query or not query.message: return
    try:
        await query.answer("✨ New Sonar conversation started.") 
    except Exception as e:
        logger.warning(f"Error answering sonar callback query for chat {query.message.chat_id}: {e}")
    await _start_new_conversation(query.message.chat_id, context, MODEL_SONAR, "Sonar")

async def new_deep_research_callback(update: Update, context: CallbackContext):
    query = update.callback_query
    if not query or not query.message: return
    try:
        await query.answer("✨ New Sonar Deep Research conversation started.") 
    except Exception as e:
        logger.warning(f"Error answering deep research callback query for chat {query.message.chat_id}: {e}")
    await _start_new_conversation(query.message.chat_id, context, MODEL_DEEP_RESEARCH, "Deep Research")

def start_bot():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CallbackQueryHandler(new_sonar_callback, pattern="^new_sonar$"))
    application.add_handler(CallbackQueryHandler(new_deep_research_callback, pattern="^new_deep_research$"))

    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    logger.info("Bot is starting...")
    application.run_polling()

if __name__ == '__main__':
    start_bot()

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(async_http_client.aclose())
        else:
            asyncio.run(async_http_client.aclose())
        logger.info("HTTP client closed.")
    except Exception as e:
        logger.error(f"Error closing HTTP client: {e}")
