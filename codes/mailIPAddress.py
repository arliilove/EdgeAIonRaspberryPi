#实现树莓派开机自动将IP地址发送到邮箱

import os
from email.mime.text import MIMEText
import smtplib
from email.header import Header

cmd = 'ifconfig'
m = os.popen(cmd)
t = m.read()
m.close()
msg = MIMEText(t,'plain','utf-8')
msg['From'] = 'Raspberry'
msg['To'] = 'destination'
msg['subject'] = Header('IP Address Report','utf-8').encode()

#请填写发件邮箱
from_add = 'XXXX@sjtu.edu.cn'
#请填写收件邮箱
to_add = 'XXXX@XXXX'

#请填写发件邮箱账号
username = 'XXXX'
#请填写发件邮箱密码
password = 'XXXX'

server = smtplib.SMTP('mail.sjtu.edu.cn:25')
#server.set_debuglevel(1)
server.login(username,password)
server.sendmail(from_add,[to_add],msg.as_string())
server.quit()

'''
文件保存后（以保存在/home/pi/Documents目录下为例）在命令中端执行如下命令：
chmod 755 /home/pi/Documents/mailIPAddress.py
sudo nano /etc/rc.local

这时，rc.local文件被打开，在 exit0 之前添加如下语句：
sleep 10
sudo python3 /home/pi/Documents/mailIPAddress.py

'''
