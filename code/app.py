#!/usr/bin/env python
import json
import os
import os.path
import shutil
import sys

from flask import current_app, Flask, jsonify, render_template, request
from flask.views import MethodView
import pickle
from gensim.models import Word2Vec
import numpy as np
from scipy import spatial
from collections import Counter
import re
import PyPDF2
import time
# Meta
##################
__version__ = '0.1.0'

# Config
##################
DEBUG = True
SECRET_KEY = 'development key'

BASE_DIR = os.path.dirname(__file__)

MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
UPLOAD_DIRECTORY = os.path.join(MEDIA_ROOT, 'upload')
CHUNKS_DIRECTORY = os.path.join(MEDIA_ROOT, 'chunks')

app = Flask(__name__)
app.config.from_object(__name__)

def findNearestResume(arrResume):
    try:
        t1 = time.time()
        t = os.path.getmtime('intermediate.txt')
        if t1-t > 30:
            os.remove('intermediate.txt')
        else:
            file = open('intermediate.txt', 'w')
            text = file.read()
            arrResume = arrResume+str(text)
            file.write(arrResume)
    except:
        file = open('intermediate.txt', 'w')
        file.write(str(arrResume))
        pass
    with open('resume_to_vector_.txt', 'rb') as handle:
        doc2vec = pickle.load(handle)
    model = Word2Vec.load('/Users/pradhumanswami/Desktop/w2vectors.txt')
    uploadedResume = arrResume
    uploadedResumeData = list(set(uploadedResume.lower().split()))
    resume_vector = np.zeros(100)
    for word in uploadedResumeData:
        word = word.decode('UTF-8')
        if word in model.vocab.keys():
            resume_vector = np.add(resume_vector, model[word])
    topDicts = dict()
    for key, value in doc2vec.items():
        if(len(value)<30):
            continue
        topDicts[key] = spatial.distance.cosine(value,resume_vector)
    sortedArr = Counter(topDicts)
    counter = 0

    answers = {}
    for k,v in sortedArr.most_common(50):
        match = re.search(r'[\w\.-]+@[\w\.-]+',k)
        try:
            answers[counter] = [topDicts[k],k,match.group(0)]
            counter = counter+1
            if(counter>10):
                break
        except:
            pass
    print(answers)
    print(len(answers))
    return json.dumps(answers)

# Utils
##################
def make_response(status=200, content=None):
    """ Construct a response to an upload request.
    Success is indicated by a status of 200 and { "success": true }
    contained in the content.

    Also, content-type is text/plain by default since IE9 and below chokes
    on application/json. For CORS environments and IE9 and below, the
    content-type needs to be text/html.
    """
    return current_app.response_class(json.dumps(content,
        indent=None if request.is_xhr else 2), mimetype='text/plain')


def validate(attrs):
    """ No-op function which will validate the client-side data.
    Werkzeug will throw an exception if you try to access an
    attribute that does not have a key for a MultiDict.
    """
    try:
        #required_attributes = ('qquuid', 'qqfilename')
        #[attrs.get(k) for k,v in attrs.items()]
        return True
    except Exception as e:
        return False


def handle_delete(uuid):
    """ Handles a filesystem delete based on UUID."""
    location = os.path.join(app.config['UPLOAD_DIRECTORY'], uuid)
    print(uuid)
    print(location)
    shutil.rmtree(location)

def handle_upload(f, attrs):
    """ Handle a chunked or non-chunked upload.
    """
    #text = handle_pdf(f)
    chunked = False
    data = f.read()
    result = findNearestResume(data)
    return result
    # dest_folder = os.path.join(app.config['UPLOAD_DIRECTORY'], attrs['qquuid'])
    #
    #
    # dest = os.path.join(dest_folder, attrs['qqfilename'])
    #
    #
    # # Chunked
    # if 'qqtotalparts' in attrs.keys() and int(attrs['qqtotalparts']) > 1:
    #     chunked = True
    #     dest_folder = os.path.join(app.config['CHUNKS_DIRECTORY'], attrs['qquuid'])
    #     dest = os.path.join(dest_folder, attrs['qqfilename'], str(attrs['qqpartindex']))
    #
    # save_upload(f, dest)
    #
    # if chunked and (int(attrs['qqtotalparts']) - 1 == int(attrs['qqpartindex'])):
    #
    #     combine_chunks(attrs['qqtotalparts'],
    #         attrs['qqtotalfilesize'],
    #         source_folder=os.path.dirname(dest),
    #         dest=os.path.join(app.config['UPLOAD_DIRECTORY'], attrs['qquuid'],
    #             attrs['qqfilename']))
    #
    #     shutil.rmtree(os.path.dirname(os.path.dirname(dest)))


def save_upload(f, path):
    """ Save an upload.
    Uploads are stored in media/uploads
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'wb+') as destination:
        destination.write(f.read())
    file = open(path, 'r')
    file.read()


def combine_chunks(total_parts, total_size, source_folder, dest):
    """ Combine a chunked file into a whole file again. Goes through each part
    , in order, and appends that part's bytes to another destination file.

    Chunks are stored in media/chunks
    Uploads are saved in media/uploads
    """

    if not os.path.exists(os.path.dirname(dest)):
        os.makedirs(os.path.dirname(dest))

    with open(dest, 'wb+') as destination:
        for i in range(int(total_parts)):
            part = os.path.join(source_folder, str(i))
            with open(part, 'rb') as source:
                destination.write(source.read())


# Views
##################
@app.route("/")
def index():
    """ The 'home' page. Returns an HTML page with Fine Uploader code
    ready to upload. This HTML page should contain your client-side code
    for instatiating and modifying Fine Uploader.
    """
    try:
        os.remove('intermediate.txt')
    except:
        pass
    return render_template('fine_uploader/index.html')

def handle_pdf(f):
    read_pdf = PyPDF2.PdfFileReader(f)
    number_of_pages = read_pdf.getNumPages()
    page = read_pdf.getPage(0)
    page_content = page.extractText()
    return page_content.encode('utf-8')

def handle_doc():
    pass


class UploadAPI(MethodView):
    """ View which will handle all upload requests sent by Fine Uploader.

    Handles POST and DELETE requests.
    """

    def post(self):
        """A POST request. Validate the form and then handle the upload
        based ont the POSTed data. Does not handle extra parameters yet.
        """
        if validate(request.form):
            res = handle_upload(request.files['qqfile'], request.form)
            return make_response(200, { "success": True, "data": res})
        else:
            return make_response(400, { "error", "Invalid request" })

    def delete(self, uuid):
        """A DELETE request. If found, deletes a file with the corresponding
        UUID from the server's filesystem.
        """
        try:
            handle_delete(uuid)
            return make_response(200, { "success": True })
        except Exception as e:
            return make_response(400, { "success": False, "error": e.message })

upload_view = UploadAPI.as_view('upload_view')
app.add_url_rule('/upload', view_func=upload_view, methods=['POST','GET'])
app.add_url_rule('/upload/<uuid>', view_func=upload_view, methods=['DELETE',])


# Main
##################
def main():
    app.run('0.0.0.0')
    return 0

if __name__ == '__main__':
    status = main()
    sys.exit(status)
