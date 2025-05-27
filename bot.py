import time
import asyncio
import telegram
import logging
import os
import sys
from datetime import datetime, timedelta
from analyzer import scan_all_crypto_symbols

# ������� logging
logging.basicConfig(
level=logging.INFO,
format="%(asctime)s - %(levelname)s - %(message)s",
handlers=[
logging.FileHandler("bot.log", encoding="utf-8"),
logging.StreamHandler()
],
force=True
)

# ������� ���� �����
BOT_TOKEN = "7914790659:AAGQenCqPQQyRUbBJDR0CH_wG_s7rluT2_I"
CHAT_ID = 632886964
LOCK_FILE = "bot.lock"
bot = telegram.Bot(token=BOT_TOKEN)

# ����� ���� ��� ���� �� ��� ���� ��� �� ��������
def check_already_running():
if os.path.exists(LOCK_FILE):
try:
with open(LOCK_FILE, "r") as f:
content = f.read().strip()
if not content:
logging.warning("���� ��� ���� ��ʡ ��� ������ ����� �����.")
remove_lock()
return
pid, timestamp = content.split(":")
timestamp = datetime.fromisoformat(timestamp)
if datetime.now() - timestamp > timedelta(hours=1):
logging.warning("���� ��� ����� ��� (��� �� 1 ����)� ��� ������ ����� �����.")
remove_lock()
return
# ����� ���� ��� PID ���� ���� ���
try:
os.kill(int(pid), 0)
logging.error(f"���� �� PID {pid} �� ��� ������.")
sys.exit(1)
except (ProcessLookupError, OSError):
logging.warning(f"PID {pid} ��� ���� ���ʡ ��� ������ ����� �����.")
remove_lock()
return
except Exception as e:
logging.error(f"��� �� ����� ���� ���: {e}")
remove_lock()
return
with open(LOCK_FILE, "w") as f:
f.write(f"{os.getpid()}:{datetime.now().isoformat()}")
logging.info(f"���� ��� �� PID {os.getpid()} � ���� {datetime.now().isoformat()} ����� ��.")

# ��� ���� ���
def remove_lock():
if os.path.exists(LOCK_FILE):
os.remove(LOCK_FILE)
logging.info("���� ��� ��� ��.")

# ���� ����� �������
async def send_signals():
logging.info("���� �Ә� �����...")

# ����� ���� ����� ���� ������� �� ���� ���� ����
try:
await bot.send_message(chat_id=CHAT_ID, text="���� ���� �� �� ���� 07:40 AM EEST, Saturday, May 17, 2025.")
except Exception as e:
logging.error(f"����� ���� ���� ������: {e}")
# ����� ���� �� ��� ����

# ���� ���� ������ ������� � ����� �� �����
async def on_signal(signal):
try:
entry = float(signal["���� ����"])
tp = float(signal["��� ���"])
sl = float(signal["�� ���"])
trade_type = signal["��� ������"]

msg = f"""?? ����� {trade_type.upper()}

����: {signal['����']}
��������: {signal['��������']}
���� ����: {entry}
��� ���: {tp}
�� ���: {sl}
��� �������: {signal['��� �������']}%
������: {signal['������']}
���� �����: {signal['���� �����']}
��Ә �� ������: {signal['��Ә �� ������']}

����� ʘ����:
{signal['�����']}

��������� �����:
{signal['���������']}

���� �����:
{signal['���� �����']}

����� ����������:
{signal['����������']}
"""
await bot.send_message(chat_id=CHAT_ID, text=msg)
logging.info(f"����� ���� {signal['����']} @ {signal['��������']} ����� ��.")
except Exception as e:
logging.error(f"����� ����� {signal.get('����', 'Unknown')} ������: {e}")

# �Ә� ����� � ����� �������
try:
await scan_all_crypto_symbols(on_signal=on_signal)
except Exception as e:
logging.error(f"��� �� �Ә� �����: {e}")

# ���� ���� ���� ����� ����
async def main():
while True:
try:
await send_signals()
logging.info("�Ә� ����� ʘ��� �ϡ �� ������ �Ә� ���� (5 �����)...")
await asyncio.sleep(300) # ������ 5 �����
except Exception as e:
logging.error(f"��� �� ���� ����: {e}")
await asyncio.sleep(60) # �� ���� ��ǡ 1 ����� ��� ���� � ������ ���� ����

# ����� ����
if __name__ == "__main__":
logging.info("���� ���� �� ���� 07:40 AM EEST, Saturday, May 17, 2025...")
check_already_running()
try:
asyncio.run(main())
except KeyboardInterrupt:
logging.info("���� ���� ����� ����� ��.")
except Exception as e:
logging.error(f"���� ��������� �� ����� ����: {e}")
finally:
remove_lock()
logging.info("���� ����� ��.")
