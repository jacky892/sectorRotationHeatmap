from datalib.commonUtil import commonUtil as cu
from bot.simpleMQUser import simpleMQUser as smu
import os

def run_job_by_qname_inline(qmsg, qname='sectorq'):
    qmsg1=f'%s' % qmsg
    qmsg=qmsg1.replace("'", '').replace('"', '')
    userid=qmsg.split('|')[0]
    msg=qmsg.split('|')[1]
    cmd=msg.split(' ')[0]
    cmd='se'
    ticker=msg.split(' ')[1]
    ticker=ticker.strip()
    print(f'focus ticker is x{ticker}')
    from datalib.commonUtil import commonUtil as cu
    from workflow.sectorRotationTraderWorkflow import sectorRotationTraderWorkflow as srt
    perf_df, trade_df, pred_df=srt.run_chatbot_sector_pred_for_ticker(ticker)
    print('returned perf_df',perf_df)
    print('pred_df ',pred_df)
    save_dict={}
    if perf_df is None:
        save_dict['userid']=userid
        save_dict['msg']=f'error with data for {ticker}'
    else:
        send_df, ofname=srt.extract_pred_table_for_sector_rotation(pred_df, ticker)
        save_dict['send_df']=send_df
        save_dict['img_list']=[ofname]
        save_dict['qmsg']=qmsg
        save_dict['userid']=userid
    pkl_fname=f'{qname}/{userid}.{cmd}.pkl.gz'
    cu.pickle_output_dict_with_img(save_dict, pkl_fname)

    return save_dict

def reply_func(txt, qname):
    txt='%s' % txt
    outdict=run_job_by_qname_inline( txt, qname=qname)
    print(outdict.keys())
    return outdict

def run_main(qname='sectorRotationBot',rqname='tgDictReply'):
    from bot.simpleMQUser import simpleMQUser as smu
    channel=smu.connect_channel(qname)
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    for method_frame, properties, body in channel.consume(qname):
        print(f'len:%s, head:%s' % (len(body), body[:20]))
        channel.basic_ack(delivery_tag = method_frame.delivery_tag)
        obj2=reply_func(body, qname)
        userid='nulluser'
        if 'userid' in obj2.keys():
            userid='%s' % obj2['userid'].replace('b', '')
        pklfname=f'tmp/ref_{userid}.{qname}.pkl.gz'
        b64=cu.pickle_output_dict_with_img(obj2, pklfname)
        smu.connect_and_post_obj(b64, rqname)
        print('obj message posted to %s %s  ' % (rqname, obj2.keys()))

run_main(qname='sectorRotationBot',rqname='tgDictReply')
