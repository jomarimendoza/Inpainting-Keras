from PIL import Image
import argparse, os, sys
from glob import glob
from os.path import join, dirname, basename

parser = argparse.ArgumentParser()

parser.add_argument('-P','--png_path', type=str, default='png_images/1.png',
                    help='png file')
parser.add_argument('-J','--jpg_path', type=str, default=None,
                    help='png file')

FLAGS = parser.parse_args()

def png_to_jpg(png,jpg):
    """ Converts a png image to jpg image """
    im = Image.open(png)
    rgb_im = im.convert('RGB')
    rgb_im.save(jpg, quality=100) # saves the image
    print('Created:', jpg )

def main():
    if os.path.isdir(FLAGS.png_path):
        # png_path is a FOLDER
        if FLAGS.jpg_path:
            if not os.path.exists( FLAGS.jpg_path ):
                print( FLAGS.jpg_path, 'created\n' )
                os.makedirs( FLAGS.jpg_path )
            pngSet = glob( join( FLAGS.png_path, '*.png' ))
            for png_path in pngSet:
                name = basename( png_path ).split('.')[0] + '.jpg' # name of file only
                jpg_path = join (FLAGS.jpg_path, name)
                png_to_jpg( png_path, jpg_path )
        else:
            # no jpg folder input
            print('Error: Input a folder name')
            sys.exit(1)
    else:
        # png_path is ONE IMAGE
        if FLAGS.jpg_path:
            # Create jpg as indicated
            png_to_jpg( FLAGS.png_path, FLAGS.jpg_path )
        else:
            # Create jpg same name as png
            name = basename( FLAGS.png_path ).split('.')[0] + '.jpg' # name of file only
            jpg_path = join( dirname( FLAGS.png_path ), name)
            png_to_jpg( FLAGS.png_path, jpg_path )

if __name__ == '__main__':
    main()
