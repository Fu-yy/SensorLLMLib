import os
import smtplib
from email.mime.text import MIMEText

import os
import ssl
import smtplib
from email.mime.text import MIMEText

def send_email_ssl465(host, port, user, password, to_email, subject, body):
    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = user
    msg["To"] = to_email
    msg["Subject"] = subject

    ctx = ssl.create_default_context()
    server = smtplib.SMTP_SSL(host, port, context=ctx, timeout=30)
    server.set_debuglevel(1)  # 打印 SMTP 过程，便于定位（稳定后改回 0）

    try:
        server.ehlo()
        server.login(user, password)
        server.sendmail(user, [to_email], msg.as_string())
    finally:
        # QQ 有时会在 quit 阶段断连接，导致 b'\x00\x00\x00'
        try:
            server.quit()
        except Exception:
            server.close()

if __name__ == "__main__":
    smtp_host = "smtp.qq.com"
    smtp_port = 465
    user = "328330246@qq.com"
    password = "ouzizmrqkwixbjef"   # 注意：这是授权码，不是登录密码
    to_email = user

    subject = os.environ.get("SMTP_SUBJECT", "[OK] NPZ build done")
    body = os.environ.get("SMTP_BODY", "NPZ build finished.")

    send_email_ssl465(smtp_host, smtp_port, user, password, to_email, subject, body)
