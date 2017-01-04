# encoding: utf8
from __future__ import unicode_literals
import codecs
import numpy

def make_bow( src_name, hist_name, dict_name ):
    word_dic = []

    # 各行を単語に分割
    lines = []
    for line in codecs.open( src_name, "r", "sjis" ).readlines():
        # 改行コードを削除
        line = line.rstrip("\r\n")

        # 単語分割
        words = line.split(" ")

        lines.append( words )

    # 単語辞書とヒストグラムを作成
    for words in lines:
        for w in words:
            # 単語がなければ辞書に追加
            if not w in word_dic:
                word_dic.append( w )

    # ヒストグラム化
    hist = numpy.zeros( (len(lines), len(word_dic)) )
    for d,words in enumerate(lines):
        for w in words:
            idx = word_dic.index(w)
            hist[d,idx] += 1


    numpy.savetxt( hist_name, hist, fmt=str("%d") )
    codecs.open( dict_name, "w", "sjis" ).write( "\n".join( word_dic ) )


def main():
    make_bow( "text.txt", "histogram_w.txt", "word_dic.txt" )

if __name__ == '__main__':
    main()