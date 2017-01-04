# encoding: shift_jis
import numpy
import random
import math
import pylab
import pickle
import os

# ハイパーパラメータ
__alpha = 1.0
__beta = 1.0

def plot( n_dz, liks, D, K ):
    print "対数尤度：", liks[-1]
    doc_dopics = numpy.argmax( n_dz , 1 )
    print "分類結果：", doc_dopics
    print "---------------------"


    # グラフ表示
    pylab.clf()
    pylab.subplot("121")
    pylab.title( "P(z|d)" )
    pylab.imshow( n_dz / numpy.tile(numpy.sum(n_dz,1).reshape(D,1),(1,K)) , interpolation="none" )
    pylab.subplot("122")
    pylab.title( "liklihood" )
    pylab.plot( range(len(liks)) , liks )
    pylab.draw()
    pylab.pause(0.1)

def calc_lda_param( docs_mdn, topics_mdn, K, dims ):
    M = len(docs_mdn)
    D = len(docs_mdn[0])

    # 各物体dにおいてトピックzが発生した回数
    n_dz = numpy.zeros((D,K))

    # 各トピックzにぴおいて特徴wが発生した回数
    n_mzw = [ numpy.zeros((K,dims[m])) for m in range(M)]

    # 各トピックが発生した回数
    n_mz = [ numpy.zeros(K) for m in range(M) ]

    # 数え上げる
    for d in range(D):
        for m in range(M):
            if dims[m]==0:
                continue
            N = len(docs_mdn[m][d])    # 物体に含まれる特徴数
            for n in range(N):
                w = docs_mdn[m][d][n]       # 物体dのn番目の特徴のインデックス
                z = topics_mdn[m][d][n]     # 特徴に割り当てれれているトピック
                n_dz[d][z] += 1
                n_mzw[m][z][w] += 1
                n_mz[m][z] += 1

    return n_dz, n_mzw, n_mz


def sample_topic( d, w, n_dz, n_zw, n_z, K, V ):
    P = [ 0.0 ] * K

    # 累積確率を計算
    P = (n_dz[d,:] + __alpha )*(n_zw[:,w] + __beta) / (n_z[:] + V *__beta)
    for z in range(1,K):
        P[z] = P[z] + P[z-1]

    # サンプリング
    rnd = P[K-1] * random.random()
    for z in range(K):
        if P[z] >= rnd:
            return z



# 単語を一列に並べたリスト変換
def conv_to_word_list( data ):
    V = len(data)
    doc = []
    for v in range(V):  # v:語彙のインデックス
        for n in range(data[v]): # 語彙の発生した回数文forを回す
            doc.append(v)
    return doc

# 尤度計算
def calc_liklihood( data, n_dz, n_zw, n_z, K, V  ):
    lik = 0

    P_wz = (n_zw.T + __beta) / (n_z + V *__beta)
    for d in range(len(data)):
        Pz = (n_dz[d] + __alpha )/( numpy.sum(n_dz[d]) + K *__alpha )
        Pwz = Pz * P_wz
        Pw = numpy.sum( Pwz , 1 ) + 0.000001
        lik += numpy.sum( data[d] * numpy.log(Pw) )

    return lik

def save_model( save_dir, n_dz, n_mzw, n_mz, M, dims ):
    try:
        os.mkdir( save_dir )
    except:
        pass

    Pdz = n_dz + __alpha
    Pdz = (Pdz.T / Pdz.sum(1)).T
    numpy.savetxt( os.path.join( save_dir, "Pdz.txt" ), Pdz, fmt=str("%f") )

    for m in range(M):
        Pwz = (n_mzw[m].T + __beta) / (n_mz[m] + dims[m] *__beta)
        Pdw = Pdz.dot(Pwz.T)
        numpy.savetxt( os.path.join( save_dir, "Pmdw[%d].txt" % m ) , Pdw )

    with open( os.path.join( save_dir, "model.pickle" ), "wb" ) as f:
        pickle.dump( [n_mzw, n_mz], f )


def load_model( load_dir ):
    model_path = os.path.join( load_dir, "model.pickle" )
    with open(model_path, "rb" ) as f:
        a,b = pickle.load( f )

    return a,b

# ldaメイン
def mlda( data, K, num_itr=100, save_dir="model", load_dir=None ):
    pylab.ion()

    # 尤度のリスト
    liks = []

    M = len(data)       # モダリティ数

    dims = []
    for m in range(M):
        if data[m] is not None:
            dims.append( len(data[m][0]) )
            D = len(data[m])    # 物体数
        else:
            dims.append( 0 )

    # data内の単語を一列に並べる（計算しやすくするため）
    docs_mdn = [[ None for i in range(D) ] for m in range(M)]
    topics_mdn = [[ None for i in range(D) ] for m in range(M)]
    for d in range(D):
         for m in range(M):
            if data[m] is not None:
                docs_mdn[m][d] = conv_to_word_list( data[m][d] )
                topics_mdn[m][d] = numpy.random.randint( 0, K, len(docs_mdn[m][d]) ) # 各単語にランダムでトピックを割り当てる

    # LDAのパラメータを計算
    n_dz, n_mzw, n_mz = calc_lda_param( docs_mdn, topics_mdn, K, dims )

    # 認識モードの時は学習したパラメータを読み込み
    if load_dir:
        n_mzw, n_mz = load_model( load_dir )

    for it in range(num_itr):
        # メインの処理
        for d in range(D):
            for m in range(M):
                if data[m] is None:
                    continue

                N = len(docs_mdn[m][d]) # 物体dのモダリティmに含まれる特徴数
                for n in range(N):
                    w = docs_mdn[m][d][n]       # 特徴のインデックス
                    z = topics_mdn[m][d][n]     # 特徴に割り当てられているカテゴリ


                    # データを取り除きパラメータを更新
                    n_dz[d][z] -= 1

                    if not load_dir:
                        n_mzw[m][z][w] -= 1
                        n_mz[m][z] -= 1

                    # サンプリング
                    z = sample_topic( d, w, n_dz, n_mzw[m], n_mz[m], K, dims[m] )

                    # データをサンプリングされたクラスに追加してパラメータを更新
                    topics_mdn[m][d][n] = z
                    n_dz[d][z] += 1

                    if not load_dir:
                        n_mzw[m][z][w] += 1
                        n_mz[m][z] += 1

        lik = 0
        for m in range(M):
            if data[m] is not None:
                lik += calc_liklihood( data[m], n_dz, n_mzw[m], n_mz[m], K, dims[m] )
        liks.append( lik )
        plot( n_dz, liks, D, K )

    save_model( save_dir, n_dz, n_mzw, n_mz, M, dims )

    pylab.ioff()
    pylab.show()

def main():
    data = []
    data.append( numpy.loadtxt( "histogram_v.txt" , dtype=numpy.int32) )
    data.append( numpy.loadtxt( "histogram_w.txt" , dtype=numpy.int32)*5 )
    mlda( data, 3, 100, "learn_result" )

    data[1] = None
    mlda( data, 3, 10, "recog_result" , "learn_result" )


if __name__ == '__main__':
    main()

