import telebot
TOKEN = '711538635:AAFzf5TcOfMgsNO_jHWy2OxK3mzoNPCqDcs'
CHATID = 9403251
enabled = True
tb = telebot.TeleBot(TOKEN)

def send_message(text):
        if not enabled:
            return
        try:
            tb.send_message(CHATID,text)
        except Exception as e:
            print(e)
        except (KeyboardInterrupt,SystemExit):
            raise
def send_photo(img):
    if not enabled:
        return
    try:
        tb.send_photo(CHATID,img)
    except Exception as e:
        print(e)
    except (KeyboardInterrupt,SystemExit):
        raise

@tb.message_handler(commands=["listen"])
def handle_listen(message):
    enabled = True

@tb.message_handler(commands=["stop"])
def handle_stop(message):
    enabled = False
