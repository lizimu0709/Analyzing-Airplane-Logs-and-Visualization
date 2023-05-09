
SECRET_KEY = "asdfasdfjasdfjasd;lf"

# Database Config
HOSTNAME = '127.0.0.1'
PORT = '3306'
DATABASE = 'boeing'
USERNAME = 'root'
PASSWORD = 'rootroot'
DB_URI = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME,PASSWORD,HOSTNAME,PORT,DATABASE)
SQLALCHEMY_DATABASE_URI = DB_URI


# Email Config
MAIL_SERVER = "smtp.gmail.com"
MAIL_USE_SSL = True
MAIL_PORT = 465
MAIL_USERNAME = "boeingloganalysis"
MAIL_PASSWORD = "dhqeltmrlmmuszpv"
MAIL_DEFAULT_SENDER = "boeingloganalysis@gmail.com"


