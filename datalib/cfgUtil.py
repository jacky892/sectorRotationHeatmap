import configparser

def get_token_input():
    cfg_fname=r"configfile.ini"
    token=input('please input tg token from /botFather e.g., 0123456789:AAABBBCCCDDDEEEFFFGGGHHHIIIJJJKKKLLLMMMNNNOOO')
    ### regex to check telegram token
    import re
    if not re.match(r'^\d{10}:[A-Za-z0-9_-]{35}$', token):
        print('error in token:, should be like 0123456789:AAABBBCCCDDDEEEFFFGGGHHHIIIJJJKKKLLLMMMNNNOOO')
        return None

    config = configparser.ConfigParser()
    # Add the structure to the file we will create
    config.add_section('telegram')
    config.set('telegram', 'token', token)
    # Write the new structure to the new file
    with open(cfg_fname, 'w') as configfile:
        config.write(configfile)
        
def check_config():
    import os
    cfg_fname="configfile.ini"
    if not os.path.exists(cfg_fname):
        get_token_input()
        
    try:
        import configparser
        config = configparser.ConfigParser()
        # Add the structure to the file we will create
        config.read(cfg_fname)

        tg_param=config['telegram']
        token=tg_param['token']
    except:
        get_token_input()

class cfgUtil:

    @staticmethod
    def get_token_from_ini():
        check_config()
        cfg_fname="configfile.ini"
        config = configparser.ConfigParser()
        # Add the structure to the file we will create
        config.read(cfg_fname)

        tg_param=config['telegram']
        token=tg_param['token']
        return token

