from PIL import Image, ImageFile, ExifTags
from hashlib import sha1
import time
import os

MAX_IMAGE_SIZE = 750, 3000

#we allowed the uploading image with the extensions 'png', 'jpg', 'jepg' and 'gif'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER = os.getcwd() + "/images"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def get_save_path(filename):
    return os.path.join(UPLOAD_FOLDER, filename)


def get_img_dir_path():
    return os.path.join(UPLOAD_FOLDER)


def get_normalized_image(data):
    image_parser = ImageFile.Parser()
    image = None
    try:
        image_parser.feed(data)
        image = image_parser.close()
        orientation = 0

        for orient in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orient] == 'Orientation':
                orientation = orient
                break

        e = image._getexif()  # returns None if no EXIF data
        if e is not None:
            exif = dict(e.items())
            orientation = exif[orientation]

            if orientation == 3:
                image = image.transpose(Image.ROTATE_180)
            elif orientation == 6:
                image = image.transpose(Image.ROTATE_270)
            elif orientation == 8:
                image = image.transpose(Image.ROTATE_90)

    except Exception as e:
        print(e)

    image.thumbnail(MAX_IMAGE_SIZE, Image.ANTIALIAS)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def generate_image_name(image_data):
    sha1sum = sha1(image_data).hexdigest()
    return '{0}.png'.format(sha1sum + str(time.time()))
