import queue                                                                                                                                                                
import threading
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from datalib.commonUtil import commonUtil as cu
import dataframe_image as dfi
import matplotlib
matplotlib.use('TkAgg')
import multiprocessing
import time
import random
from PIL import Image, ImageChops
from tkinter import *


import socket
import gzip,pickle,os,sys,time,base64

def get_help_message():
    help_msg='to get vcp plot: v AAPL, v CBA.ax, v ETH=C, v AUDUSD=X\n to get sector heatmap: se XLF, se CBA.ax, sc ETH=C, sf AUDUSD=X"
    return help_msg

''' run jobs that show up in the rabbitmq with queue name {qname} '''
def run_job_by_qname_inline(userid, msq, qname='sectorq'):
    '''
    sample message is '{userid}|se AAPL' with the second part being the ticker
    results is than stor in a pickle file with name {qname}/{userid}.se.pkl.gz
    which is then base64 encoded, posted to rabbitmq and read by the tgbot and sent to the user  

    '''
    cmd=msq.split(' ')[0]
    ticker=msq.split(' ')[1]
    subcmd='e'
    if len(cmd)>1:
        subcmd=cmd[1]
    peer_dict={}
    peer_dict['e']='etf'
    peer_dict['c']='cmc'
    peer_dict['f']='fx'
    peer=peer_dict[subcmd]
    from datalib.commonUtil import commonUtil as cu
    from workflow.sectorRotationTraderWorkflow import sectorRotationTraderWorkflow as srt
    perf_df, trade_df, pred_df=srt.run_chatbot_sector_pred_for_ticker(ticker,peer)
    send_df, ofname=srt.extract_pred_table_for_sector_rotation(pred_df, ticker)
    save_dict={}
    save_dict['send_df']=send_df
    save_dict['ofname']=ofname
    save_dict['msg']=None
    pkl_fname=f'{qname}/{user_id}.{cmd}.pkl.gz'
    cu.pickle_output_dict_with_img(save_dict, pkl_fname)
    
    return pkl_fname

'''redirect commeand to different message queues based on the first word of the message'''
def sample_router(msg, job_dict=None):                                                    
                                                                                      
    job_dict
    default_qname='jobq'
    if job_dict is None:
        g_job_dict={}
        g_job_dict['sectorRotationBot']=['se', 's']
        g_job_dict['vcp']=['vc', 'v', 'vcp']        
    else:
        g_job_dict=job_dict
    cmd=msg.split(' ')[0]
    print('got message:', '%s cmd:%s'  % (msg, cmd))
    for k in g_job_dict:
        cmd_list=g_job_dict[k]
        print('cmd_list vs k:%s %s ' % (cmd_list, k))
        if cmd.lower() in cmd_list:
            return k
    return default_qname


            
def tg_poll(tgbot):
    import time                                                                                                                                                             
    import pandas as pd
    print('now in tgbot poll')
    while True:
        ts1=pd.to_datetime('today')
        print(f'my poll:{ts1}')
        tgbot.polling(skip_pending=True, interval=5, timeout=10, long_polling_timeout=20)
    #    tgbot.polling()
        ts2=pd.to_datetime('today')
        print(f'my poll ed {ts2}')
        time.sleep(2)

        
def try_get_reply_b64_obj_msg(tgbot, rqname='tgDictReply'):
    from bot.simpleMQUser import simpleMQUser as smu
    channel=smu.connect_channel(rqname)

    func=reply_tg_func
    #simpleMQUser.try_read_txt_msg(channel, rqname, reply_func)
    from bot.simpleMQUser import simpleMQUser as smu
    for method_frame, properties, body in channel.consume(rqname):
        print(f'len:%s, head:%s' % (len(body), body[:20]))
        encoded_string=body
        obj2=base64.b64decode(encoded_string)
        if not os.path.exists('tmp'):
            os.makedirs('tmp')
        tmpfname='tmp/tmp.pkl.gz'
        with gzip.open(tmpfname, 'wb') as f:
            f.write(obj2)
        obj2=cu.load_output_dict_pickle_with_img(tmpfname)
        print(obj2.keys())
        channel.basic_ack(delivery_tag = method_frame.delivery_tag)
        reply_tg_func(tgbot, obj2)
        #b64_objmsg = base64.b64encode(obj2)
        #smu.connect_and_post_obj(b64_objmsg, f'{qname}')

def loop_reply_job(tgbot, qname='tgDictReply'):     
    print(' replyq tgbot is ',tgbot)
    import time
    import pandas as pd
    run_cnt=0
    while True:
        print(f'{qname} start loop ',pd.to_datetime('today'))
        time.sleep(5)
        try_get_reply_b64_obj_msg(tgbot, qname)
        run_cnt=run_cnt+1
        print( f' run_cnt:{run_cnt} {qname} end loop ',pd.to_datetime('today'))
        if run_cnt>10: 
            import os, sys
            print(f"RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR restart after cnount is {run_cnt}")
            os.execv(sys.argv[0], sys.argv)        



#Create a window
if cu.is_console_mode():           
    window=Tk()            


def reply_tg_func(tgbot, obj):
    import os,pickle,gzip
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    print(obj.keys())
    cnt=0;
    if not 'userid' in obj.keys():
        print('missing user id: %s' % obj.keys())
        return None
    userid=obj['userid']
    if 'b' in '%s' % userid:
        userid=int(userid.replace('b', ''))
    print('user id is ',userid)
    if 'img_obj' in obj.keys():
        img_list=obj['img_obj']
        for im in img_list:
            ofname=f'tmp/{cnt}.png'
            bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
            diff = ImageChops.difference(im, bg)
            diff = ImageChops.add(diff, diff, 2.0, -100)
            bbox = diff.getbbox()
            if bbox:
                im=im.crop(bbox)
            im.save(ofname)
            tgbot.send_photo(userid, photo=open(ofname, 'rb'))
    msg_txt=''
    if 'msg' in obj.keys():
        msg_txt=obj['msg']
        print('also have txt msg:',msg_txt)
    if len(msg_txt)>0:
        print(f'sending msg:{msg_txt}')
        tgbot.send_message(userid, msg_txt)





def main_run():
    import telebot
    hostname = socket.gethostname()
    from datalib.cfgUtil import cfgUtil
    token=cfgUtil.get_token_from_ini()
    token=cfgUtil.get_token_from_ini()

    if 'tgbot' in dir():
        del tgbot
    tgbot = telebot.TeleBot(token)        
    g_router_func=None
    g_job_dict=None
    g_use_rbmq=False 
    g_use_rbmq=True
    param={}

    @tgbot.message_handler(func=lambda message: True)
    def message_handler(message, router_func=None):
        from bot.simpleMQUser import  simpleMQUser as smu
        userid=message.from_user.id
        tgmsg=message
        username=message.from_user.first_name
        print('incoming user id is ',userid, username, router_func)
        if not f'{userid}' in ['1266302555', '1703257834', '5054619528']:
            tgbot.reply_to(message, f'unauthorized user {username}')
            return
        print('from_user:',message.from_user)
        if 'help' in message.text.lower():
            msg=get_help_message()
            tgbot.reply_to(message, msg)
        else:
            tgbot.reply_to(message, f'u{userid} {username} %s ' % message.text)
        msg=message.text.strip()
        print('got message:', '%s %s' % (hostname, msg))
        if router_func is None:
            g_router_func=sample_router
        else:
            g_router_func=router_func
        qname=g_router_func(msg, g_job_dict)
        print(f'router_func is {router_func}, qname is {qname}')

        if g_use_rbmq:
            print('using mq')
            smu.connect_and_post_txtmsg(f'{userid}|{msg}', qname)

        else:
            print('not using mq')
            ofname, print_df=run_job_by_qname_inline(userid, msg, qname)
            if not print_df is None:
                tgbot.send_photo(userid, photo=open(ofname, 'rb'))
        return tgbot
    
    params=[tgbot]    
    if g_use_rbmq:
        print('start reply q also')
        t3=threading.Thread(target=loop_reply_job, args=(params))
        t3.start()    

    t2=threading.Thread(target=tg_poll, args=(params))
    t2.start()
    t2.join()
    print('ok')
    
if __name__=='__main__':
    main_run()
