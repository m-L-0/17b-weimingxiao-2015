import os

from flask import Flask, render_template,url_for, session,redirect
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from t import cnn
app = Flask(__name__)
app.config['SECRET_KEY'] = 'I have a dream'
app.config['UPLOADED_PHOTOS_DEST'] = '/home/r/captcha/load/'
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB


class UploadForm(FlaskForm):
    photo = FileField(validators=[
        FileAllowed(photos, u'只能上传图片！'), 
        FileRequired(u'文件未选择！')])
    submit = SubmitField(u'上传')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = photos.url(filename)
        path = '/home/r/captcha/load/'+filename
        print(path)
        session['result'] = cnn(path)
        session['img'] = file_url
        return redirect(url_for('upload_file'))
    else:
        file_url = None
    # session['result'] = ''
    return render_template('b.html', form=form, file_url=session.get('img'),result=session.get('result'))


if __name__ == '__main__':
    app.run()
