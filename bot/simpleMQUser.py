from datalib.commonUtil import commonUtil as cu

class simpleMQUser:

    @staticmethod
    def connect_channel(qname):
        import pika
        username=cu.getProp('rq_username')
#        username='tgbot'
#        passwd='tgbot'
        passwd=cu.getProp('rq_password')    
        rbmq_ip=cu.getProp('rbmq_ip')        
#        rbmq_ip='127.0.0.1'
        credentials = pika.PlainCredentials(username, passwd)

        connection = pika.BlockingConnection(pika.ConnectionParameters(rbmq_ip, credentials=credentials))
        channel = connection.channel()
        return channel

    @staticmethod
    def connect_and_post_txtmsg(msg, qname, close_channl=False):
        channel=simpleMQUser.connect_channel(qname)
        channel.basic_publish(exchange='', routing_key=qname, body=msg)
        if close_channl:
            channel.close()
        return channel
            
    @staticmethod
    def connect_and_post_obj(b64_objmsg, qname, close_channl=False):
        import pika
        username=cu.getProp('rq_username')
#        username='tgbot'
#        passwd='tgbot'
        passwd=cu.getProp('rq_password')    
        rbmq_ip=cu.getProp('rbmq_ip')        
#        rbmq_ip='127.0.0.1'
        print('connect info:',username, rbmq_ip)
        credentials = pika.PlainCredentials(username, passwd)
        connection = pika.BlockingConnection(pika.ConnectionParameters(rbmq_ip, credentials=credentials))
        channel = connection.channel()
        channel.basic_publish(exchange='', routing_key=qname, body=b64_objmsg)
        if close_channl:
            channel.close()
        return channel

            
    @staticmethod
    def try_read_txt_msg(channel, qname, func):
        for method_frame, properties, body in channel.consume(qname):
            print(f'len:%s, head:%s' % (len(body), body[:20]))
            outdict=func(body)
            
    @staticmethod
    def try_read_obj_msg(channel, qname, func):
        import base64
        for method_frame, properties, body in channel.consume(qname):
            print(f'len:%s, head:%s' % (len(body), body[:20]))
            encoded_string=body
            bytedata=base64.b64decode(encoded_string)
            tmpfname='tmp.pkl.gz'
            with gzip.open(tmpfname, 'wb') as f:
                f.write(bytedata)
            obj_msg=cu.load_output_dict_pickle_with_img(tmpfname)
            outdict=func(obj_msg)
    
    @staticmethod    
    def reply_func(obj_msg):
        print(obj_msg.keys())
        print('got key already!')


def testrun_obj_msg(pklfname='ref_userid.pkl.gz'):

    qname='sectorRotationBot'
    obj_dict=cu.load_output_dict_pickle_with_img(pklfname)
    with gzip.open(pklfname) as f:
        b64_objmsg = base64.b64encode(f.read())
    channel=simpleMQUser.connect_and_post_obj_msg(b64_objmsg, qname)

def testrun_txt_msg(txt='se XLK'):
    qname='sectorRotationBot'
    channel=simpleMQUser.connect_and_post_txtmsg(txt, qname)
    
#testrun_txt_msg()
