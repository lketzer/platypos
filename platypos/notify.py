### notify_message.py ###

import os
from io import StringIO
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart

def message(subject="Python Notification", text="", img=None, attachment=None):
    # build message contents
    msg = MIMEMultipart()
    msg['Subject'] = subject  # add in the subject
    msg.attach(MIMEText(text))  # add text contents

    # check if we have anything given in the img parameter
    if img is not None:
        # if we do, we want to iterate through the images, so let's check that
        # what we have is actually a list
        if type(img) is not list:
            img = [img]  # if it isn't a list, make it one
        # now iterate through our list
        for one_img in img:
            img_data = open(one_img, 'rb').read()  # read the image binary data
            # attach the image data to MIMEMultipart using MIMEImage, we add
            # the given filename use os.basename
            msg.attach(MIMEImage(img_data, name=os.path.basename(one_img)))

    # we do the same for attachments as we did for images
    if attachment is not None:
        if type(attachment) is not list:
            attachment = [attachment]  # if it isn't a list, make it one
        with open(attachment, 'rb') as f:
            # read in the attachment using MIMEApplication
            file = MIMEApplication(
                f.read(),
                name=os.path.basename(attachment)
            )
        # here we edit the attached file metadata
        file['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment)}"'
        msg.attach(file)  # finally, add the attachment to our message object
    return msg


### notify_smtp ###

import smtplib
import socket
from io import StringIO

def send(msg, email, pwd, server='smtp.gmail.com', port=587):
    # contain following in try-except in case of momentary network errors
    try:
        # initialise connection to email server, the default is Outlook
        smtp = smtplib.SMTP(server, port)
        # this is the 'Extended Hello' command, essentially greeting our SMTP or ESMTP server
        smtp.ehlo()
        # this is the 'Start Transport Layer Security' command, tells the server we will
        # be communicating with TLS encryption
        smtp.starttls()
        
        # read email and password from file
#         with open(path_to_username_password_files+'/email.txt', 'r') as fp:
#             email = fp.read()
#         with open(path_to_username_password_files+'/password.txt', 'r') as fp:
#             pwd = fp.read()
            
        # login to the email server
        smtp.login(email, pwd)
        # send notification to self
        smtp.sendmail(email, email, msg.as_string())
        # disconnect from the server
        smtp.quit()
    except socket.gaierror:
        print("Network connection error, email not sent.")