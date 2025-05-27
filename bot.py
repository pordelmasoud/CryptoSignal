import time
import asyncio
import telegram
import logging
import os
import sys
from datetime import datetime, timedelta
from analyzer import scan_all_crypto_symbols

# ÊäÙíãÇÊ logging
logging.basicConfig(
level=logging.INFO,
format="%(asctime)s - %(levelname)s - %(message)s",
handlers=[
logging.FileHandler("bot.log", encoding="utf-8"),
logging.StreamHandler()
],
force=True
)

# ÊäÙíãÇÊ ÑÈÇÊ ÊáÑÇã
BOT_TOKEN = "7914790659:AAGQenCqPQQyRUbBJDR0CH_wG_s7rluT2_I"
CHAT_ID = 632886964
LOCK_FILE = "bot.lock"
bot = telegram.Bot(token=BOT_TOKEN)

# ÈÑÑÓí Çíä˜å ÂíÇ ÑÈÇÊ ÏÑ ÍÇá ÇÌÑÇ ÇÓÊ ÈÇ ÒãÇäÈäÏí
def check_already_running():
if os.path.exists(LOCK_FILE):
try:
with open(LOCK_FILE, "r") as f:
content = f.read().strip()
if not content:
logging.warning("İÇíá Şİá ÎÇáí ÇÓÊ¡ Şİá äÇÏíÏå ÑİÊå ãíÔæÏ.")
remove_lock()
return
pid, timestamp = content.split(":")
timestamp = datetime.fromisoformat(timestamp)
if datetime.now() - timestamp > timedelta(hours=1):
logging.warning("İÇíá Şİá ŞÏíãí ÇÓÊ (ÈíÔ ÇÒ 1 ÓÇÚÊ)¡ Şİá äÇÏíÏå ÑİÊå ãíÔæÏ.")
remove_lock()
return
# ÈÑÑÓí Çíä˜å ÂíÇ PID åäæÒ İÚÇá ÇÓÊ
try:
os.kill(int(pid), 0)
logging.error(f"ÑÈÇÊ ÈÇ PID {pid} ÏÑ ÍÇá ÇÌÑÇÓÊ.")
sys.exit(1)
except (ProcessLookupError, OSError):
logging.warning(f"PID {pid} ÏíÑ İÚÇá äíÓÊ¡ Şİá äÇÏíÏå ÑİÊå ãíÔæÏ.")
remove_lock()
return
except Exception as e:
logging.error(f"ÎØÇ ÏÑ ÈÑÑÓí İÇíá Şİá: {e}")
remove_lock()
return
with open(LOCK_FILE, "w") as f:
f.write(f"{os.getpid()}:{datetime.now().isoformat()}")
logging.info(f"İÇíá Şİá ÈÇ PID {os.getpid()} æ ÒãÇä {datetime.now().isoformat()} ÇíÌÇÏ ÔÏ.")

# ÍĞİ İÇíá Şİá
def remove_lock():
if os.path.exists(LOCK_FILE):
os.remove(LOCK_FILE)
logging.info("İÇíá Şİá ÍĞİ ÔÏ.")

# ÊÇÈÚ ÇÑÓÇá ÓíäÇáåÇ
async def send_signals():
logging.info("ÔÑæÚ ÇÓ˜ä ÈÇÒÇÑ...")

# ÇÑÓÇá íÇã Çæáíå ÈÑÇí ÇØãíäÇä ÇÒ İÚÇá ÈæÏä ÑÈÇÊ
try:
await bot.send_message(chat_id=CHAT_ID, text="ÑÈÇÊ İÚÇá ÔÏ ÏÑ ÓÇÚÊ 07:40 AM EEST, Saturday, May 17, 2025.")
except Exception as e:
logging.error(f"ÇÑÓÇá íÇã ÊÓÊí äÇãæİŞ: {e}")
# ÇÏÇãå ÏÇÏä Èå ÌÇí ÊæŞİ

# ÊÇÈÚ ÈÑÇí ãÏíÑíÊ ÓíäÇáåÇ æ ÇÑÓÇá Èå ÊáÑÇã
async def on_signal(signal):
try:
entry = float(signal["ŞíãÊ æÑæÏ"])
tp = float(signal["åÏİ ÓæÏ"])
sl = float(signal["ÍÏ ÖÑÑ"])
trade_type = signal["äæÚ ãÚÇãáå"]

msg = f"""?? ÓíäÇá {trade_type.upper()}

äãÇÏ: {signal['äãÇÏ']}
ÊÇíãİÑíã: {signal['ÊÇíãİÑíã']}
ŞíãÊ æÑæÏ: {entry}
åÏİ ÓæÏ: {tp}
ÍÏ ÖÑÑ: {sl}
ÓØÍ ÇØãíäÇä: {signal['ÓØÍ ÇØãíäÇä']}%
ÇãÊíÇÒ: {signal['ÇãÊíÇÒ']}
ŞÏÑÊ ÓíäÇá: {signal['ŞÏÑÊ ÓíäÇá']}
ÑíÓ˜ Èå ÑíæÇÑÏ: {signal['ÑíÓ˜ Èå ÑíæÇÑÏ']}

ÊÍáíá Ê˜äí˜Çá:
{signal['ÊÍáíá']}

ÑæÇäÔäÇÓí ÈÇÒÇÑ:
{signal['ÑæÇäÔäÇÓí']}

ÑæäÏ ÈÇÒÇÑ:
{signal['ÑæäÏ ÈÇÒÇÑ']}

ÊÍáíá İÇäÏÇãäÊÇá:
{signal['İÇäÏÇãäÊÇá']}
"""
await bot.send_message(chat_id=CHAT_ID, text=msg)
logging.info(f"ÓíäÇá ÈÑÇí {signal['äãÇÏ']} @ {signal['ÊÇíãİÑíã']} ÇÑÓÇá ÔÏ.")
except Exception as e:
logging.error(f"ÇÑÓÇá ÓíäÇá {signal.get('äãÇÏ', 'Unknown')} äÇãæİŞ: {e}")

# ÇÓ˜ä ÈÇÒÇÑ æ ÇÑÓÇá ÓíäÇáåÇ
try:
await scan_all_crypto_symbols(on_signal=on_signal)
except Exception as e:
logging.error(f"ÎØÇ ÏÑ ÇÓ˜ä ÈÇÒÇÑ: {e}")

# ÊÇÈÚ ÇÕáí ÈÑÇí ÇÌÑÇí ÑÈÇÊ
async def main():
while True:
try:
await send_signals()
logging.info("ÇÓ˜ä ÈÇÒÇÑ Ê˜ãíá ÔÏ¡ ÏÑ ÇäÊÙÇÑ ÇÓ˜ä ÈÚÏí (5 ÏŞíŞå)...")
await asyncio.sleep(300) # ÇäÊÙÇÑ 5 ÏŞíŞå
except Exception as e:
logging.error(f"ÎØÇ ÏÑ ÍáŞå ÇÕáí: {e}")
await asyncio.sleep(60) # ÏÑ ÕæÑÊ ÎØÇ¡ 1 ÏŞíŞå ÕÈÑ ˜äíÏ æ ÏæÈÇÑå ÊáÇÔ ˜äíÏ

# ÇÌÑÇí ÑÈÇÊ
if __name__ == "__main__":
logging.info("ÔÑæÚ ÑÈÇÊ ÏÑ ÓÇÚÊ 07:40 AM EEST, Saturday, May 17, 2025...")
check_already_running()
try:
asyncio.run(main())
except KeyboardInterrupt:
logging.info("ÑÈÇÊ ÊæÓØ ˜ÇÑÈÑ ãÊæŞİ ÔÏ.")
except Exception as e:
logging.error(f"ÎØÇí ÛíÑãäÊÙÑå ÏÑ ÇÌÑÇí ÑÈÇÊ: {e}")
finally:
remove_lock()
logging.info("ÑÈÇÊ ãÊæŞİ ÔÏ.")
