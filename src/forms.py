import wtforms
from wtforms.validators import Email, Length, EqualTo, InputRequired
from models import UserModel, EmailCaptchaModel
from exts import db


class RegisterForm(wtforms.Form):
    email = wtforms.StringField(validators=[Email(message="Invalid email format!")])
    captcha = wtforms.StringField(validators=[Length(min=4, max=4, message="Invalid captcha format!")])
    username = wtforms.StringField(validators=[Length(min=3, max=20, message="Invalid username format!")])
    password = wtforms.StringField(validators=[Length(min=6, max=20, message="Invalid password format!")])
    password_confirm = wtforms.StringField(validators=[EqualTo("password", message="Passwords do not match!")])

    # Custom validation:
    # 1. Check if email is already registered
    def validate_email(self, field):
        email = field.data
        user = UserModel.query.filter_by(email=email).first()
        if user:
            raise wtforms.ValidationError(message="Email already registered!")

    # 2. Check if captcha is correct
    def validate_captcha(self, field):
        captcha = field.data
        email = self.email.data
        captcha_model = EmailCaptchaModel.query.filter_by(email=email, captcha=captcha).first()
        if not captcha_model:
            raise wtforms.ValidationError(message="Invalid email or captcha!")


class LoginForm(wtforms.Form):
    email = wtforms.StringField(validators=[Email(message="Invalid email format!")])
    password = wtforms.StringField(validators=[Length(min=6, max=20, message="Invalid password format!")])