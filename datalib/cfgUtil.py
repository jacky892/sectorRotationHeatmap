import configparser

def get_token_input():
    cfg_fname=r"configfile.ini"
    token=input('please input tg token from /botFather')
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

