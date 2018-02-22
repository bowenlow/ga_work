from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField
from wtforms.validators import DataRequired

class LoginForm(FlaskForm):
	username = StringField('UserName', validators=[DataRequired()])
	password = PasswordField('Password', validators=[DataRequired()])
	remember_me = BooleanField('Remember me')
	submit = SubmitField('Sign In')


class ReviewForm(FlaskForm):
	reviewtext = TextAreaField('Review',validators=[DataRequired()])
	submit = SubmitField('Submit2')
