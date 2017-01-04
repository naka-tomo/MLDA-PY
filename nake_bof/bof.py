# encoding: utf8
from __future__ import unicode_literals
import cv2
import numpy
import glob

_detecotor = cv2.AKAZE_create()
def calc_feature( filename ):
    img = cv2.imread( filename, 0 )
    kp, discriptor = _detecotor.detectAndCompute(img,None)
    return numpy.array(discriptor, dtype=numpy.float32 )

# コードブックを作成
def make_codebook( images, code_book_size, save_name ):
    bow_trainer = cv2.BOWKMeansTrainer( code_book_size )

    for img in images:
        f = calc_feature(img)  # 特徴量計算
        bow_trainer.add( f )

    code_book = bow_trainer.cluster()
    numpy.savetxt( save_name, code_book )

# ヒストグラム作成
def make_bof( code_book_name, images, hist_name ):
    code_book = numpy.loadtxt( code_book_name, dtype=numpy.float32 )

    knn= cv2.ml.KNearest_create()
    knn.train(code_book, cv2.ml.ROW_SAMPLE, numpy.arange(len(code_book),dtype=numpy.float32))

    hists = []
    for img in images:
        f = calc_feature( img )
        idx = knn.findNearest( f, 1 )[1]

        h = numpy.zeros( len(code_book) )
        for i in idx:
            h[int(i)] += 1

        hists.append( h )

    numpy.savetxt( hist_name, hists, fmt=str("%d")  )


def main():
    files = glob.glob("images/*.png")
    make_codebook( files, 50, "codebook.txt" )
    make_bof( "codebook.txt", files, "histogram_v.txt" )

if __name__ == '__main__':
    main()